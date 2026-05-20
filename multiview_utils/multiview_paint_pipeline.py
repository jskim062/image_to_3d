import os
import sys
import copy
import torch
import trimesh
import numpy as np
from PIL import Image

# Bind hy3dpaint path to Python path
hy3dpaint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Hunyuan3D-2.1", "hy3dpaint"))
if hy3dpaint_path not in sys.path:
    sys.path.insert(0, hy3dpaint_path)

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from utils.uvwrap_utils import mesh_uv_wrap
from utils.simplify_mesh_utils import remesh_mesh
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
from multiview_utils.image_utils import align_multiview_colors

class MultiViewPaintPipeline(Hunyuan3DPaintPipeline):
    """
    Upgraded texturing pipeline that accepts 4 to 6 real aligned photographs
    and projects them directly onto the generated 3D mesh with seam inpainting.
    """
    
    @torch.no_grad()
    def __call__(self, mesh_path=None, image_path=None, output_mesh_path=None, use_remesh=True, save_glb=True, views=None):
        """
        Infers seamless high-fidelity texture maps using 4 to 6 real photos.
        - views: list of PIL Images in order: [Front, Left, Back, Right, (Top, Bottom)]
        """
        if views is None or len(views) < 4:
            print("[multiview] Warning: No multi-view images supplied or fewer than 4. Falling back to single-image pipeline.")
            return super().__call__(mesh_path, image_path, output_mesh_path, use_remesh, save_glb)
            
        print(f"[multiview] Starting Multi-View texture synthesis with {len(views)} real photos...")
        
        # 1. Pre-process and equalize lighting across all views using our histogram matching
        print("[multiview] Aligning illumination profiles across all views...")
        views = align_multiview_colors(views)
        
        # 2. Process and remesh geometry
        path = os.path.dirname(mesh_path)
        if use_remesh:
            processed_mesh_path = os.path.join(path, "white_mesh_remesh.obj")
            remesh_mesh(mesh_path, processed_mesh_path)
        else:
            processed_mesh_path = mesh_path
            
        if output_mesh_path is None:
            output_mesh_path = os.path.join(path, "textured_mesh.obj")
            
        # 3. Load 3D mesh and perform UV unwrapping
        mesh = trimesh.load(processed_mesh_path)
        mesh = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh=mesh)
        
        # 4. View selection (will select the 6 candidate orthogonal cameras)
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num,
        )
        
        # 5. Render normal and position maps for geometry mapping
        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)
        
        # 6. Prepare prompt image (Front view)
        front_image = views[0].convert("RGB")
        image_style = [front_image]
        
        # 7. Generate base PBR material maps using diffusion
        print("[multiview] Generating baseline PBR material structures...")
        multiviews_pbr = self.models["multiview_model"](
            image_style,
            normal_maps + position_maps,
            prompt="high quality, professional game asset, realistic textures",
            custom_view_size=self.config.resolution,
            resize_input=True,
        )
        
        # 8. Enhance and super-resolve generated maps
        enhance_images = {}
        enhance_images["albedo"] = copy.deepcopy(multiviews_pbr["albedo"])
        enhance_images["mr"] = copy.deepcopy(multiviews_pbr["mr"])
        
        for i in range(len(enhance_images["albedo"])):
            enhance_images["albedo"][i] = self.models["super_model"](enhance_images["albedo"][i])
            enhance_images["mr"][i] = self.models["super_model"](enhance_images["mr"][i])
            
        # 9. Overwrite the generated albedo maps with our aligned real photos!
        print(f"[multiview] Overwriting {len(views)} views with 100% accurate real-world photographs...")
        for i in range(min(len(views), len(enhance_images["albedo"]))):
            # Resize the real aligned photo to match the target render resolution (typically 2048)
            real_photo = views[i].resize((self.config.render_size, self.config.render_size), Image.Resampling.LANCZOS)
            enhance_images["albedo"][i] = real_photo
            
        # 10. Standardize dimensions for baking
        for i in range(len(enhance_images["albedo"])):
            if i >= len(views):
                # Ensure any non-overwritten views (e.g. top/bottom in 4-view mode) are correctly sized
                enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                    (self.config.render_size, self.config.render_size)
                )
            enhance_images["mr"][i] = enhance_images["mr"][i].resize(
                (self.config.render_size, self.config.render_size)
            )
            
        # 11. Project and bake Albedo textures
        print("[multiview] Baking high-resolution color maps onto UV charts...")
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        
        # 12. Project and bake Metallic-Roughness (material) textures
        print("[multiview] Baking PBR material maps onto UV charts...")
        texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        
        # 13. Perform dynamic inpainting on transition seams and occlusion gaps
        print("[multiview] Inpainting and blending transition boundaries...")
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture, force_set=True)
        
        texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
        self.render.set_texture_mr(texture_mr)
        
        # 14. Save OBJ mesh and convert to GLB
        print("[multiview] Finalizing model output...")
        self.render.save_mesh(output_mesh_path, downsample=True)
        
        if save_glb:
            glb_path = output_mesh_path.replace(".obj", ".glb")
            convert_obj_to_glb(output_mesh_path, glb_path)
            print(f"[SUCCESS] Ultra-fidelity multi-view GLB successfully generated at: {glb_path}")
            return glb_path
            
        return output_mesh_path
