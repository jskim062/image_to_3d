import os
import sys
import argparse
import torch
from PIL import Image

# Setup paths for Hunyuan3D-2.1 sub-modules
workspace_dir = os.path.abspath(os.path.dirname(__file__))
hy3dshape_path = os.path.join(workspace_dir, "Hunyuan3D-2.1", "hy3dshape")
hy3dpaint_path = os.path.join(workspace_dir, "Hunyuan3D-2.1", "hy3dpaint")

for path in [hy3dshape_path, hy3dpaint_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

from multiview_utils.image_utils import slice_turnaround_sheet
from multiview_utils.multiview_paint_pipeline import MultiViewPaintPipeline

def run_multiview_pipeline(args):
    print("=====================================================================")
    print("[*] Hunyuan3D Fork: 4-to-6 Directional Multi-View Image Fusion System")
    print("=====================================================================")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Slice and align turnaround sheet
    print(f"[*] Processing turnaround sheet: {args.sheet}")
    views = slice_turnaround_sheet(args.sheet, num_views=args.num_views, bg_threshold=args.bg_threshold)
    print(f"[+] Successfully sliced and standardized {len(views)} views.")
    
    # Save sliced views in their original sliced order for diagnostics
    sliced_paths = []
    for idx, view in enumerate(views):
        view_path = os.path.join(args.output_dir, f"sliced_view_{idx}.png")
        view.save(view_path)
        sliced_paths.append(view_path)
    print(f"[+] Diagnostic sliced views saved to: {args.output_dir}")
    
    # Reorder views to match Hunyuan3D camera selection order: [Front, Right, Back, Left, (Top, Bottom)]
    # Target camera mapping:
    #   Index 0: Front
    #   Index 1: Right
    #   Index 2: Back
    #   Index 3: Left
    #   Index 4: Top
    #   Index 5: Bottom
    print(f"[*] Reordering views from '{args.view_order}' to match Hunyuan3D camera selection index mapping...")
    if args.view_order == "front_left_back_right":
        if len(views) == 4:
            # Sliced: [0: Front, 1: Left, 2: Back, 3: Right]
            views = [views[0], views[3], views[2], views[1]]
        elif len(views) == 6:
            # Sliced: [0: Front, 1: Left, 2: Back, 3: Right, 4: Top, 5: Bottom]
            views = [views[0], views[3], views[2], views[1], views[4], views[5]]
    elif args.view_order == "front_right_back_left":
        if len(views) == 4:
            # Sliced: [0: Front, 1: Right, 2: Back, 3: Left]
            views = [views[0], views[1], views[2], views[3]]
        elif len(views) == 6:
            # Sliced: [0: Front, 1: Right, 2: Back, 3: Left, 4: Top, 5: Bottom]
            views = [views[0], views[1], views[2], views[3], views[4], views[5]]
    else:
        raise ValueError(f"Unknown view order: {args.view_order}")
    
    # 2. Geometry (Shape) Generation Stage
    base_mesh_path = args.mesh
    if not base_mesh_path:
        print("\n--- STAGE 1: Shape Generation (Base Geometry) ---")
        # Use the first view (Front) to generate the base 3D mesh
        front_view_path = sliced_paths[0]
        
        # Load Shape Pipeline
        print("[shape] Loading Hunyuan3D Shape Pipeline...")
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2.1',
            device=args.device,
            torch_dtype=torch.float16,
        )
        
        # Run Shape Inference
        print(f"[shape] Reconstructing 3D base mesh from Front view...")
        with torch.no_grad():
            mesh = shape_pipeline(image=front_view_path)[0]
            
        base_mesh_path = os.path.join(args.output_dir, "base_geometry.glb")
        mesh.export(base_mesh_path)
        print(f"[SUCCESS] Base geometry generated: {base_mesh_path}")
        
        # Clean VRAM from shape stage
        del mesh, shape_pipeline
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\n[*] Skipping Shape Stage: Reusing existing mesh: {base_mesh_path}")
        
    # 3. Texturing Stage using our Custom Multi-View Projection subclass
    print("\n--- STAGE 2: Multi-View Spatial Projection & Seam-Blending ---")
    
    # Load Paint Config
    from textureGenPipeline import Hunyuan3DPaintConfig
    cfg = Hunyuan3DPaintConfig(max_num_view=6, resolution=args.resolution)
    cfg.device = args.device
    cfg.realesrgan_ckpt_path = os.path.join(hy3dpaint_path, "ckpt", "RealESRGAN_x4plus.pth")
    cfg.multiview_cfg_path = os.path.join(hy3dpaint_path, "cfgs", "hunyuan-paint-pbr.yaml")
    
    # Initialize Multi-View Paint Pipeline
    print("[paint] Loading paint pipeline with multi-view capability...")
    paint_pipeline = MultiViewPaintPipeline(cfg)
    
    # Run multi-view projection texturing
    obj_output_path = os.path.join(args.output_dir, "final_multiview_result.obj")
    final_glb_path = paint_pipeline(
        mesh_path=base_mesh_path,
        image_path=sliced_paths[0],  # Dummy prompt image
        output_mesh_path=obj_output_path,
        use_remesh=True,
        save_glb=True,
        views=views,
    )
    
    print("\n=====================================================================")
    print(f"[SUCCESS] Seamless Multi-View 3D Model created: {final_glb_path}")
    print("=====================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate seamless 3D models using 4-to-6 directional views.")
    parser.add_argument("--sheet", type=str, required=True, help="Path to horizontal multi-view sheet image.")
    parser.add_argument("--num_views", type=int, default=None, help="Number of views (4 or 6). Auto-detected if omitted.")
    parser.add_argument("--view_order", type=str, default="front_left_back_right", 
                        choices=["front_left_back_right", "front_right_back_left"],
                        help="Horizontal layout view order in turnaround sheet. Default: front_left_back_right.")
    parser.add_argument("--mesh", type=str, default=None, help="Path to existing base GLB/OBJ mesh (reuses geometry).")
    parser.add_argument("--resolution", type=int, default=512, help="Texturing pipeline resolution (default: 512).")
    parser.add_argument("--bg_threshold", type=int, default=240, help="White background cropping threshold (default: 240).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target CUDA device (default: cuda:0).")
    parser.add_argument("--output_dir", type=str, default="./output_multiview", help="Directory to save finished assets.")
    
    args = parser.parse_args()
    run_multiview_pipeline(args)
