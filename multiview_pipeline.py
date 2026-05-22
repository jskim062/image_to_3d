import os
import sys
from unittest.mock import MagicMock

# Mock Blender (bpy) module globally to prevent ModuleNotFoundError on Kaggle/servers
sys.modules['bpy'] = MagicMock()

# Prevent torchvision import crash in modern environments (post-torchvision 0.15+) for basicsr/realesrgan
import torchvision.transforms.functional as _tvf
sys.modules['torchvision.transforms.functional_tensor'] = _tvf

# Globally silence tqdm progress bars to prevent notebook output flooding (OOM/long scrolls)
try:
    import tqdm
    class SilentTqdm(tqdm.tqdm):
        def __init__(self, *args, **kwargs):
            kwargs['disable'] = True
            super().__init__(*args, **kwargs)
    tqdm.tqdm = SilentTqdm
    sys.modules['tqdm'].tqdm = SilentTqdm
    for m_name in list(sys.modules.keys()):
        if m_name.startswith('tqdm'):
            m = sys.modules[m_name]
            if hasattr(m, 'tqdm'):
                setattr(m, 'tqdm', SilentTqdm)
except Exception:
    pass

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

# Prevent namespace package shadowing for custom_rasterizer in Kaggle/Linux environments
try:
    import custom_rasterizer
    if not hasattr(custom_rasterizer, 'rasterize'):
        import custom_rasterizer.custom_rasterizer as actual_cr
        sys.modules['custom_rasterizer'] = actual_cr
except ImportError:
    pass

from multiview_utils.image_utils import slice_turnaround_sheet
from multiview_utils.multiview_paint_pipeline import MultiViewPaintPipeline

# ---------------------------------------------------------------------------
# CLIP-based automatic view classification
# ---------------------------------------------------------------------------

_CLIP_PROMPTS = {
    "front":  "front view of an object or building or character, front side, facing camera",
    "back":   "back view of an object or building or character, rear side, facing away",
    "right":  "right side profile view of an object or building or character",
    "left":   "left side profile view of an object or building or character",
    "top":    "top-down bird's eye view of an object or building or character from above",
    "bottom": "bottom view of an object or building or character from below",
}

_HUNYUAN_ORDER = ["front", "right", "back", "left", "top", "bottom"]

def classify_views_with_clip(views, labels=None, device="cuda:0"):
    """
    Classifies view directions automatically using CLIP and reorders them to Hunyuan3D camera sequence.
    If text label images are present, uses zero-shot OCR matching to maximize accuracy.
    """
    print("[CLIP] Loading CLIP model for automatic view classification...")
    from transformers import CLIPProcessor, CLIPModel
    import numpy as np
    import gc

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    directions = _HUNYUAN_ORDER[:len(views)]
    vis_prompts = [_CLIP_PROMPTS[d] for d in directions]
    
    label_candidates = {
        "front":  ["front view", "full front view", "front"],
        "back":   ["back view", "full back view", "back"],
        "right":  ["right side view", "right view", "right side", "right"],
        "left":   ["left side view", "left view", "left side", "left"],
        "top":    ["top-down view", "top down view", "top view", "top"],
        "bottom": ["bottom-up view", "bottom up view", "bottom view", "bottom"],
    }

    # 모든 라벨용 프롬프트 리스트 생성
    lbl_prompts = []
    prompt_to_dir = {}
    for d in directions:
        for p in label_candidates[d]:
            lbl_prompts.append(p)
            prompt_to_dir[p] = d

    scores = np.zeros((len(views), len(directions)))
    with torch.no_grad():
        for i, view in enumerate(views):
            label_img = labels[i] if labels is not None else None
            
            # A. Visual object matching score
            vis_inputs = processor(
                text=vis_prompts,
                images=view.convert("RGB"),
                return_tensors="pt",
                padding=True,
            )
            vis_inputs = {k: v.to(device) for k, v in vis_inputs.items()}
            # Shape: (num_directions,)
            vis_scores = model(**vis_inputs).logits_per_image[0].cpu().numpy()
            
            # B. Text label OCR matching score
            if label_img is not None:
                lbl_inputs = processor(
                    text=lbl_prompts,
                    images=label_img.convert("RGB"),
                    return_tensors="pt",
                    padding=True,
                )
                lbl_inputs = {k: v.to(device) for k, v in lbl_inputs.items()}
                # Shape: (total_lbl_prompts,)
                lbl_logits = model(**lbl_inputs).logits_per_image[0].cpu().numpy()
                
                # Extract max score for each direction
                lbl_scores = np.zeros(len(directions))
                for dir_idx, direction in enumerate(directions):
                    indices = [idx for idx, p in enumerate(lbl_prompts) if prompt_to_dir[p] == direction]
                    lbl_scores[dir_idx] = np.max(lbl_logits[indices])
                
                combined = 0.85 * lbl_scores + 0.15 * vis_scores
            else:
                combined = vis_scores
                
            scores[i] = combined

    assignment = {}
    used = set()
    for dir_idx, direction in enumerate(directions):
        candidates = [i for i in range(len(views)) if i not in used]
        best_view_idx = max(candidates, key=lambda i: scores[i, dir_idx])
        assignment[direction] = views[best_view_idx]
        used.add(best_view_idx)
        print(f"  [CLIP] {direction:6s} ← view #{best_view_idx}  "
              f"(score {scores[best_view_idx, dir_idx]:.2f})")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    result = [assignment[d] for d in _HUNYUAN_ORDER if d in assignment]
    print(f"[CLIP] Classification complete. Output order: {[d for d in _HUNYUAN_ORDER if d in assignment]}")
    return result

def run_multiview_pipeline(args):
    print("=====================================================================")
    print("[*] Hunyuan3D Fork: 4-to-6 Directional Multi-View Image Fusion System")
    print("=====================================================================")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clean up old output files from previous runs in this output directory to prevent leftovers/mixing
    print(f"[*] Cleaning up previous run outputs in {args.output_dir}...")
    input_mesh_abs = os.path.abspath(args.mesh) if args.mesh else None
    for filename in os.listdir(args.output_dir):
        file_path = os.path.join(args.output_dir, filename)
        if filename.startswith("sliced_view_") and filename.endswith(".png"):
            try:
                os.remove(file_path)
            except Exception:
                pass
        elif filename in ["base_geometry.glb", "base_geometry.obj", "final_multiview_result.glb", "final_multiview_result.obj", "final_multiview_result.mtl", "textured_mesh.obj", "white_mesh_remesh.obj"]:
            if input_mesh_abs and os.path.abspath(file_path) == input_mesh_abs:
                print(f"[*] Preserving input mesh: {filename}")
                continue
            try:
                os.remove(file_path)
            except Exception:
                pass
    
    # 1. Slice and align turnaround sheet
    print(f"[*] Processing turnaround sheet: {args.sheet}")
    views, labels = slice_turnaround_sheet(
        args.sheet,
        num_views=args.num_views,
        bg_threshold=args.bg_threshold,
        return_labels=True,
    )
    print(f"[+] Successfully sliced and standardized {len(views)} views.")
    
    # Save sliced views in their original sliced order for diagnostics
    sliced_paths = []
    for idx, view in enumerate(views):
        view_path = os.path.join(args.output_dir, f"sliced_view_{idx}.png")
        view.save(view_path)
        sliced_paths.append(view_path)
    print(f"[+] Diagnostic sliced views saved to: {args.output_dir}")
    
    # Reorder views to match Hunyuan3D camera selection order: [Front, Right, Back, Left, (Top, Bottom)]
    if getattr(args, "auto_classify", False):
        print("[*] Auto-classifying view directions with CLIP using label guiding...")
        views = classify_views_with_clip(views, labels=labels, device=args.device)
        print(f"[+] CLIP classified {len(views)} views.")
    else:
        print(f"[*] Reordering views from '{args.view_order}' to match Hunyuan3D camera selection index mapping...")
        if "," in args.view_order:
            user_order = [s.strip().lower() for s in args.view_order.split(",")]
            if len(user_order) != len(views):
                raise ValueError(f"Length of custom view order ({len(user_order)}) must match number of sliced views ({len(views)}). Got: {args.view_order}")
                
            # Target directions in the exact order Hunyuan3D expects
            if len(views) == 4:
                target_dirs = ["front", "right", "back", "left"]
            else:
                target_dirs = ["front", "right", "back", "left", "top", "bottom"]
                
            reordered_views = []
            for target in target_dirs:
                if target not in user_order:
                    raise ValueError(f"Required direction '{target}' missing from custom view order: {args.view_order}")
                sliced_idx = user_order.index(target)
                reordered_views.append(views[sliced_idx])
            views = reordered_views
        elif args.view_order == "front_left_back_right":
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
                        help="View order in the sheet. Can be a preset ('front_left_back_right', 'front_right_back_left') or a custom comma-separated string (e.g. 'front,left,back,right,top,bottom').")
    parser.add_argument("--auto_classify", action="store_true",
                        help="Use CLIP to automatically detect front/back/left/right/top/bottom. Overrides --view_order.")
    parser.add_argument("--mesh", type=str, default=None, help="Path to existing base GLB/OBJ mesh (reuses geometry).")
    parser.add_argument("--resolution", type=int, default=512, help="Texturing pipeline resolution (default: 512).")
    parser.add_argument("--bg_threshold", type=int, default=240, help="White background cropping threshold (default: 240).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target CUDA device (default: cuda:0).")
    parser.add_argument("--output_dir", type=str, default="./output_multiview", help="Directory to save finished assets.")
    
    args = parser.parse_args()
    run_multiview_pipeline(args)
