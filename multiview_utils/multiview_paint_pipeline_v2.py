import os
import sys
import copy
import torch
import trimesh
import numpy as np
from PIL import Image
from unittest.mock import MagicMock

# bpy is not available on Kaggle — mock only when actually missing
try:
    import bpy  # noqa: F401
except ImportError:
    sys.modules['bpy'] = MagicMock()

# torchvision.transforms.functional_tensor removed in 0.15+; alias for basicsr/realesrgan
import torchvision.transforms.functional as _tvf
if 'torchvision.transforms.functional_tensor' not in sys.modules:
    sys.modules['torchvision.transforms.functional_tensor'] = _tvf

hy3dpaint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Hunyuan3D-2.1", "hy3dpaint"))
if hy3dpaint_path not in sys.path:
    sys.path.insert(0, hy3dpaint_path)

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from utils.uvwrap_utils import mesh_uv_wrap
from utils.simplify_mesh_utils import remesh_mesh
from multiview_utils.image_utils import align_multiview_colors


def robust_convert_obj_to_glb(obj_path, glb_path):
    is_mock = isinstance(sys.modules.get('bpy'), MagicMock)

    success = False
    if not is_mock:
        try:
            from DifferentiableRenderer.mesh_utils import convert_obj_to_glb as blender_convert
            print("[multiview] Attempting Blender-based OBJ to GLB conversion...")
            success = blender_convert(obj_path, glb_path)
        except Exception as e:
            print(f"[multiview] Blender conversion failed: {e}")

    if not success or not os.path.exists(glb_path):
        print("[multiview] Using Trimesh for OBJ to GLB conversion...")
        try:
            scene = trimesh.load(obj_path)
            scene.export(glb_path)
            success = os.path.exists(glb_path)
            if success:
                print(f"[multiview] Trimesh GLB saved: {glb_path}")
        except Exception as e:
            print(f"[multiview] Trimesh GLB conversion failed: {e}")

    return success


def _ensure_obj(mesh_path, output_dir):
    """GLB/GLTF 입력을 OBJ로 변환. 이미 OBJ면 그대로 반환."""
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext in ('.glb', '.gltf'):
        print(f"[multiview] Input mesh is {ext.upper()} — converting to OBJ for pipeline compatibility...")
        obj_path = os.path.join(output_dir, "base_geometry_converted.obj")
        try:
            m = trimesh.load(mesh_path, force='mesh')
            m.export(obj_path)
            print(f"[multiview] Converted to OBJ: {obj_path}")
            return obj_path
        except Exception as e:
            raise RuntimeError(f"GLB → OBJ 변환 실패: {e}") from e
    return mesh_path


class MultiViewPaintPipeline(Hunyuan3DPaintPipeline):
    """
    Upgraded texturing pipeline that accepts 4 to 6 real aligned photographs
    and projects them directly onto the generated 3D mesh with seam inpainting.

    v2 fixes:
    - GLB input auto-converted to OBJ before remesh (fixes line-81 crash)
    - remesh failure falls back gracefully instead of crashing
    """

    @torch.no_grad()
    def __call__(self, mesh_path=None, image_path=None, output_mesh_path=None,
                 use_remesh=True, save_glb=True, views=None):
        """
        views: list of PIL Images [Front, Right, Back, Left, (Top, Bottom)]
        """
        if views is None or len(views) < 4:
            print("[multiview] Fewer than 4 views — falling back to single-image pipeline.")
            return super().__call__(mesh_path, image_path, output_mesh_path, use_remesh, save_glb)

        print(f"[multiview] Starting Multi-View texture synthesis with {len(views)} views...")

        # 1. Color alignment across views
        print("[multiview] Aligning illumination profiles across all views...")
        views = align_multiview_colors(views)

        # 2. GLB → OBJ conversion (v2 fix: prevents remesh crash)
        output_dir = os.path.dirname(mesh_path) if mesh_path else "."
        mesh_path = _ensure_obj(mesh_path, output_dir)

        # 3. Remesh with graceful fallback
        if output_mesh_path is None:
            output_mesh_path = os.path.join(output_dir, "textured_mesh.obj")

        if use_remesh:
            processed_mesh_path = os.path.join(output_dir, "white_mesh_remesh.obj")
            try:
                remesh_mesh(mesh_path, processed_mesh_path)
            except Exception as e:
                print(f"[multiview] Warning: remesh_mesh failed ({e}). Using original mesh.")
                processed_mesh_path = mesh_path
        else:
            processed_mesh_path = mesh_path

        # 4. UV unwrap and load
        mesh = trimesh.load(processed_mesh_path)
        mesh = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh=mesh)

        # 5. View selection
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num,
        )

        # 6. Render normal and position maps
        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(
            selected_camera_elevs, selected_camera_azims
        )

        # 7. Generate base PBR maps via diffusion
        print("[multiview] Generating baseline PBR material structures...")
        front_image = views[0].convert("RGB")
        multiviews_pbr = self.models["multiview_model"](
            [front_image],
            normal_maps + position_maps,
            prompt="high quality, professional game asset, realistic textures",
            custom_view_size=self.config.resolution,
            resize_input=True,
        )

        # 8. Super-resolve PBR maps
        enhance_images = {
            "albedo": [self.models["super_model"](img) for img in multiviews_pbr["albedo"]],
            "mr":     [self.models["super_model"](img) for img in multiviews_pbr["mr"]],
        }

        # 9. Overwrite albedo slots with real photographs
        print(f"[multiview] Projecting {len(views)} real photos onto albedo slots...")
        for i in range(min(len(views), len(enhance_images["albedo"]))):
            enhance_images["albedo"][i] = views[i].resize(
                (self.config.render_size, self.config.render_size), Image.Resampling.LANCZOS
            )

        # 10. Normalize all slot sizes
        for i in range(len(enhance_images["albedo"])):
            if i >= len(views):
                enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                    (self.config.render_size, self.config.render_size)
                )
            enhance_images["mr"][i] = enhance_images["mr"][i].resize(
                (self.config.render_size, self.config.render_size)
            )

        # 11. Bake albedo
        print("[multiview] Baking color maps onto UV charts...")
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        # 12. Bake metallic-roughness
        print("[multiview] Baking PBR material maps...")
        texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        # 13. Seam inpainting
        print("[multiview] Inpainting seam boundaries...")
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture, force_set=True)

        texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
        self.render.set_texture_mr(texture_mr)

        # 14. Export
        print("[multiview] Saving output mesh...")
        self.render.save_mesh(output_mesh_path, downsample=True)

        if save_glb:
            glb_path = output_mesh_path.replace(".obj", ".glb")
            robust_convert_obj_to_glb(output_mesh_path, glb_path)
            print(f"[SUCCESS] Multi-view GLB saved: {glb_path}")
            return glb_path

        return output_mesh_path
