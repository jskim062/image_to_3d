import os
import sys
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

import gc
import argparse
import numpy as np
import torch
from PIL import Image

workspace_dir = os.path.abspath(os.path.dirname(__file__))
hy3dshape_path = os.path.join(workspace_dir, "Hunyuan3D-2.1", "hy3dshape")
hy3dpaint_path = os.path.join(workspace_dir, "Hunyuan3D-2.1", "hy3dpaint")

for path in [hy3dshape_path, hy3dpaint_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Prevent namespace package shadowing for custom_rasterizer
try:
    import custom_rasterizer
    if not hasattr(custom_rasterizer, 'rasterize'):
        import custom_rasterizer.custom_rasterizer as actual_cr
        sys.modules['custom_rasterizer'] = actual_cr
except ImportError:
    pass

from multiview_utils.image_utils import slice_turnaround_sheet
from multiview_utils.multiview_paint_pipeline_v2 import MultiViewPaintPipeline

# ---------------------------------------------------------------------------
# View reorder helpers
# ---------------------------------------------------------------------------

# Hunyuan3D 내부 카메라 인덱스 순서: [Front, Right, Back, Left, (Top, Bottom)]
_REORDER = {
    #  입력 순서              4-view 매핑               6-view 매핑
    "front_left_back_right":  ([0, 3, 2, 1],           [0, 3, 2, 1, 4, 5]),
    "front_right_back_left":  ([0, 1, 2, 3],           [0, 1, 2, 3, 4, 5]),
    "front_back_left_right":  ([0, 3, 1, 2],           [0, 3, 1, 2, 4, 5]),
}


def reorder_views(views, view_order):
    """입력 순서 문자열에 따라 Hunyuan3D 카메라 순서로 재정렬."""
    if view_order not in _REORDER:
        raise ValueError(
            f"Unknown view_order '{view_order}'. "
            f"Choices: {list(_REORDER.keys())}"
        )
    idx_4, idx_6 = _REORDER[view_order]
    indices = idx_6 if len(views) == 6 else idx_4
    reordered = [views[i] for i in indices]
    print(f"[*] View reorder ({view_order}): {list(range(len(views)))} → {indices}")
    return reordered


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

# Hunyuan3D 기대 순서
_HUNYUAN_ORDER = ["front", "right", "back", "left", "top", "bottom"]


def classify_views_with_clip(views, labels=None, device="cuda:0"):
    """
    CLIP으로 각 뷰를 front/back/left/right/(top/bottom) 자동 분류 후
    Hunyuan3D 카메라 순서 [Front, Right, Back, Left, (Top, Bottom)]로 반환.
    텍스트 라벨 이미지(labels)가 있다면 CLIP의 zero-shot OCR 매칭 성능을 사용해 방향 인식을 극대화합니다.
    """
    print("[CLIP] Loading CLIP model for automatic view classification...")
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # 뷰 수에 맞게 방향 후보 제한
    directions = _HUNYUAN_ORDER[:len(views)]
    vis_prompts = [_CLIP_PROMPTS[d] for d in directions]
    
    # 방향별 상세 텍스트 라벨 키워드 후보 (OCR 매칭용)
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

    scores = np.zeros((len(views), len(directions)))  # [num_views, num_directions]
    with torch.no_grad():
        for i, view in enumerate(views):
            label_img = labels[i] if labels is not None else None
            
            # A. 비주얼 이미지 일괄 매칭
            vis_inputs = processor(
                text=vis_prompts,
                images=view.convert("RGB"),
                return_tensors="pt",
                padding=True,
            )
            vis_inputs = {k: v.to(device) for k, v in vis_inputs.items()}
            # Shape: (num_directions,)
            vis_scores = model(**vis_inputs).logits_per_image[0].cpu().numpy()
            
            # B. 라벨 이미지 일괄 매칭 (라벨 영역이 존재할 때만)
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
                
                # 각 방향 후보별 최댓값 매칭
                lbl_scores = np.zeros(len(directions))
                for dir_idx, direction in enumerate(directions):
                    indices = [idx for idx, p in enumerate(lbl_prompts) if prompt_to_dir[p] == direction]
                    lbl_scores[dir_idx] = np.max(lbl_logits[indices])
                
                # 텍스트 정보가 매우 명확하므로 85%의 큰 가중치를 부여
                combined = 0.85 * lbl_scores + 0.15 * vis_scores
            else:
                combined = vis_scores
                
            scores[i] = combined

    # Greedy 할당: 우선순위 순으로 최고 점수 뷰를 방향에 배정
    assignment = {}
    used = set()
    for dir_idx, direction in enumerate(directions):
        candidates = [i for i in range(len(views)) if i not in used]
        best_view_idx = max(candidates, key=lambda i: scores[i, dir_idx])
        assignment[direction] = views[best_view_idx]
        used.add(best_view_idx)
        print(f"  [CLIP] {direction:6s} ← view #{best_view_idx}  "
              f"(score {scores[best_view_idx, dir_idx]:.2f})")

    # VRAM 해제
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Hunyuan3D 순서로 반환
    result = [assignment[d] for d in _HUNYUAN_ORDER if d in assignment]
    print(f"[CLIP] Classification complete. Output order: "
          f"{[d for d in _HUNYUAN_ORDER if d in assignment]}")
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_multiview_pipeline(args):
    print("=====================================================================")
    print("[*] Hunyuan3D Fork v2: Multi-View Image Fusion System")
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

    # 1. Slice turnaround sheet
    print(f"[*] Processing turnaround sheet: {args.sheet}")
    views, labels = slice_turnaround_sheet(
        args.sheet,
        num_views=args.num_views,
        bg_threshold=args.bg_threshold,
        use_rembg=args.use_rembg,
        return_labels=True,
    )
    print(f"[+] Sliced {len(views)} views.")

    # 진단용 저장
    sliced_paths = []
    for idx, view in enumerate(views):
        p = os.path.join(args.output_dir, f"sliced_view_{idx}.png")
        view.save(p)
        sliced_paths.append(p)
    print(f"[+] Diagnostic sliced views saved to: {args.output_dir}")

    # 2. 방향 정렬 (CLIP 자동 분류 또는 수동 순서 재배열)
    if args.auto_classify:
        print("[*] Auto-classifying view directions with CLIP using label guiding...")
        views = classify_views_with_clip(views, labels=labels, device=args.device)
        print(f"[+] CLIP classified {len(views)} views.")
    else:
        print(f"[*] Reordering views from '{args.view_order}' to Hunyuan3D order...")
        views = reorder_views(views, args.view_order)

    # 3. Shape Generation
    base_mesh_path = args.mesh
    if not base_mesh_path:
        print("\n--- STAGE 1: Shape Generation ---")
        front_view_path = sliced_paths[0]

        print("[shape] Loading Hunyuan3D Shape Pipeline...")
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2.1',
            device=args.device,
            torch_dtype=torch.float16,
        )

        print(f"[shape] Reconstructing base mesh from Front view "
              f"(octree={args.octree_resolution}, steps={args.steps}, cfg={args.guidance_scale})...")
        with torch.no_grad():
            mesh = shape_pipeline(
                image=front_view_path,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                octree_resolution=args.octree_resolution,
            )[0]

        base_mesh_path = os.path.join(args.output_dir, "base_geometry.glb")
        exported = mesh.export(base_mesh_path)
        # trimesh 일부 버전은 경로를 줘도 bytes를 반환하고 쓰지 않음 → 수동 저장
        if not os.path.exists(base_mesh_path):
            if isinstance(exported, bytes) and len(exported) > 0:
                with open(base_mesh_path, 'wb') as _f:
                    _f.write(exported)
                print("[shape] trimesh returned bytes; written to disk manually.")
            else:
                raise RuntimeError(
                    f"[shape] mesh.export() failed to create {base_mesh_path}"
                )
        print(f"[SUCCESS] Base geometry saved: {base_mesh_path}")

        del mesh, shape_pipeline
        gc.collect()
        torch.cuda.empty_cache()

        if args.shape_only:
            print("[*] --shape_only 모드: Shape 생성 완료 후 종료합니다.")
            return
    else:
        print(f"\n[*] Skipping Shape Stage: reusing {base_mesh_path}")

    # 4. Texturing
    print("\n--- STAGE 2: Multi-View Spatial Projection & Seam-Blending ---")

    from textureGenPipeline import Hunyuan3DPaintConfig
    cfg = Hunyuan3DPaintConfig(max_num_view=6, resolution=args.resolution)
    cfg.device = args.device
    cfg.realesrgan_ckpt_path = os.path.join(hy3dpaint_path, "ckpt", "RealESRGAN_x4plus.pth")
    cfg.multiview_cfg_path = os.path.join(hy3dpaint_path, "cfgs", "hunyuan-paint-pbr.yaml")

    print("[paint] Loading MultiViewPaintPipeline v2...")
    paint_pipeline = MultiViewPaintPipeline(cfg)

    obj_output_path = os.path.join(args.output_dir, "final_multiview_result.obj")
    final_glb_path = paint_pipeline(
        mesh_path=base_mesh_path,
        image_path=sliced_paths[0],
        output_mesh_path=obj_output_path,
        use_remesh=True,
        save_glb=True,
        views=views,
    )

    print("\n=====================================================================")
    print(f"[SUCCESS] Final 3D model: {final_glb_path}")
    print("=====================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hunyuan3D v2: Multi-view turnaround → textured 3D GLB"
    )
    parser.add_argument("--sheet", type=str, required=True,
                        help="Path to horizontal turnaround sheet image.")
    parser.add_argument("--num_views", type=int, default=None,
                        help="Number of views (4 or 6). Auto-detected if omitted.")
    parser.add_argument("--view_order", type=str, default="front_left_back_right",
                        choices=list(_REORDER.keys()),
                        help=(
                            "Horizontal layout order of views in turnaround sheet. "
                            "Ignored when --auto_classify is set. "
                            f"Choices: {list(_REORDER.keys())}. "
                            "Default: front_left_back_right."
                        ))
    parser.add_argument("--auto_classify", action="store_true",
                        help="Use CLIP to automatically detect front/back/left/right. "
                             "Overrides --view_order.")
    parser.add_argument("--mesh", type=str, default=None,
                        help="Path to existing GLB/OBJ mesh (skips shape generation).")
    # ── 형상 품질 ──────────────────────────────────────────────────────────────
    parser.add_argument("--octree_resolution", type=int, default=512,
                        help="메쉬 디테일 해상도. 기본 384→512. T4 권장 최대: 640. "
                             "높을수록 디테일↑, VRAM↑, 속도↓.")
    parser.add_argument("--steps", type=int, default=50,
                        help="디퓨전 inference step 수. 기본 50. 100이면 더 정밀하나 2배 느림.")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier-free guidance scale. 기본 5.0. 높을수록 이미지 충실도↑.")
    # ── 입력 품질 ──────────────────────────────────────────────────────────────
    parser.add_argument("--use_rembg", action="store_true",
                        help="AI 배경 제거(rembg) 사용. threshold 방식보다 외곽선이 정밀함.")
    parser.add_argument("--bg_threshold", type=int, default=240,
                        help="threshold 방식 흰 배경 제거 임계값. --use_rembg 비활성 시 적용.")
    # ── 공통 ───────────────────────────────────────────────────────────────────
    parser.add_argument("--shape_only", action="store_true",
                        help="Shape 생성만 수행하고 base_geometry.glb 저장 후 종료. "
                             "VRAM 완전 해제 후 텍스처 단계를 별도 프로세스로 실행할 때 사용.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Texturing pipeline resolution (default: 512).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device (default: cuda:0).")
    parser.add_argument("--output_dir", type=str, default="./output_multiview",
                        help="Output directory.")

    args = parser.parse_args()
    run_multiview_pipeline(args)
