#!/usr/bin/env python
import os
import sys
import argparse
import gc
import torch
import numpy as np
import trimesh
from PIL import Image

# Ensure workspace and Hunyuan3D-Shape directories are in system path
workspace_dir = os.path.abspath(os.path.dirname(__file__))
hy3dshape_path = os.path.join(workspace_dir, "Hunyuan3D-2.1", "hy3dshape")

for path in [workspace_dir, hy3dshape_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import turnaround slicing and CAD exporter
from multiview_utils.image_utils import slice_turnaround_sheet
from glb_to_cad import convert_glb_to_cad

# ---------------------------------------------------------------------------
# Core Geometric Operators
# ---------------------------------------------------------------------------

def apply_ground_locking(mesh, ratio=0.05, target_z=0.0):
    """
    Locates vertices in the lowest 'ratio' fraction of the mesh height, 
    flattens them to the minimum Z coordinate, and translates the entire mesh 
    so the bottom is exactly at target_z (usually 0.0 for ground plane).
    
    This preserves watertightness since topology is untouched.
    """
    vertices = mesh.vertices.copy()
    if len(vertices) == 0:
        return mesh
        
    z_coords = vertices[:, 2]
    z_min = float(np.min(z_coords))
    z_max = float(np.max(z_coords))
    z_height = z_max - z_min
    
    if z_height <= 1e-6:
        print("[Ground Locking] Mesh has zero vertical height. Skipping.")
        return mesh
        
    z_threshold = z_min + z_height * ratio
    
    # Identify and flatten bottom-most vertices
    bottom_mask = z_coords <= z_threshold
    num_flattened = np.sum(bottom_mask)
    
    vertices[bottom_mask, 2] = z_min
    mesh.vertices = vertices
    
    # Translate mesh so that z_min is shifted to target_z
    translation = target_z - z_min
    translation_vector = np.array([0.0, 0.0, translation])
    mesh.vertices = mesh.vertices + translation_vector
    
    print(f"[+] Ground Locking Applied:")
    print(f"    - Vertices flattened to flat base: {num_flattened} / {len(vertices)} ({num_flattened/len(vertices)*100:.1f}%)")
    print(f"    - Original Z span: [{z_min:.4f}, {z_max:.4f}]")
    print(f"    - New Z span: [{target_z:.4f}, {(z_max + translation):.4f}]")
    
    return mesh


def apply_mesh_smoothing(mesh, method="laplacian", iterations=10, lamb=0.5):
    """
    Applies Laplacian or volume-preserving Taubin smoothing to eliminate 
    pixelated/voxelized AI reconstruction artifacts.
    """
    if method == "none" or method is None:
        print("[Smoothing] Smoothing disabled.")
        return mesh
        
    print(f"[*] Applying {method} smoothing ({iterations} iterations)...")
    try:
        import trimesh.smoothing
        if method.lower() == "laplacian":
            trimesh.smoothing.filter_laplacian(mesh, lamb=lamb, iterations=iterations)
        elif method.lower() == "taubin":
            # Taubin requires a negative nu parameter to prevent shrinkage (usually nu = -lamb * 1.05)
            nu = -lamb * 1.05
            trimesh.smoothing.filter_taubin(mesh, lamb=lamb, nu=nu, iterations=iterations)
        else:
            print(f"[Warning] Unknown smoothing method '{method}'. Skipping.")
            return mesh
        print(f"[+] Mesh smoothing completed successfully.")
    except Exception as e:
        print(f"[Warning] Mesh smoothing failed: {e}. Proceeding with original mesh.")
        
    return mesh


def generate_contour_slices(mesh, interval=0.2, output_dir=".", base_name="contours"):
    """
    Slices the mesh at regular vertical intervals and exports:
      1. A combined 3D DXF file containing all contours at their respective elevations.
      2. A directory of 2D SVG files (one for each slice) for laser cutting or waffle grids.
    """
    if interval <= 0.0:
        print("[Slicing] Slicing interval <= 0.0. Slicing disabled.")
        return None
        
    z_coords = mesh.vertices[:, 2]
    z_min = float(np.min(z_coords))
    z_max = float(np.max(z_coords))
    
    # Generate slice heights from slightly above bottom to slightly below top
    epsilon = 1e-4
    slice_heights = np.arange(z_min + interval, z_max - epsilon, interval)
    
    if len(slice_heights) == 0:
        print(f"[Slicing] No slice heights generated. Interval {interval:.3f} exceeds Z span {z_max - z_min:.3f}.")
        return None
        
    print(f"[*] Slicing mesh horizontally at {len(slice_heights)} levels (from Z={slice_heights[0]:.2f} to Z={slice_heights[-1]:.2f}, step={interval:.2f})...")
    
    sections = []
    svg_dir = os.path.join(output_dir, f"{base_name}_svg")
    os.makedirs(svg_dir, exist_ok=True)
    
    for idx, z in enumerate(slice_heights):
        plane_origin = [0.0, 0.0, z]
        plane_normal = [0.0, 0.0, 1.0]
        
        try:
            section = mesh.section(plane_normal=plane_normal, plane_origin=plane_origin)
            if section is None:
                continue
                
            sections.append(section)
            
            # Export 2D SVG contour
            sec_2D, _ = section.to_2D()
            svg_path = os.path.join(svg_dir, f"contour_z_{z:.3f}.svg")
            sec_2D.export(svg_path)
        except Exception as e:
            # Handle potential non-intersections silently or as warnings
            pass
            
    if len(sections) == 0:
        print("[Slicing] No active mesh sections were sliced successfully.")
        return None
        
    try:
        combined = trimesh.path.util.concatenate(sections)
        dxf_path = os.path.join(output_dir, f"{base_name}_3d_contours.dxf")
        combined.export(dxf_path)
        print(f"[+] Contour Slicing Exported Successfully:")
        print(f"    - 3D DXF file (all layers): {dxf_path}")
        print(f"    - 2D SVG files directory: {svg_dir}")
        return dxf_path
    except Exception as e:
        print(f"[Error] Failed to export combined 3D DXF contours: {e}")
        return None

# ---------------------------------------------------------------------------
# Main Pipeline Function
# ---------------------------------------------------------------------------

def run_architectural_pipeline(args):
    print("=====================================================================")
    print("Hunyuan3D: Specialized Architectural CAD & Fabrication Pipeline")
    print("=====================================================================")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Handle Input Image (Slice turnaround or use single sketch directly)
    front_image_path = None
    
    if args.image:
        print(f"[*] Loading single architectural sketch: {args.image}")
        front_image_path = args.image
    elif args.sheet:
        print(f"[*] Processing turnaround sheet: {args.sheet}")
        views = slice_turnaround_sheet(
            args.sheet,
            num_views=args.num_views,
            bg_threshold=args.bg_threshold,
            use_rembg=args.use_rembg,
        )
        print(f"[+] Sliced {len(views)} views from sheet.")
        
        # Save sliced front view diagnostic
        front_view = views[0]
        front_image_path = os.path.join(args.output_dir, "sliced_front_view.png")
        front_view.save(front_image_path)
        print(f"[+] Diagnostic sliced front view saved: {front_image_path}")
    else:
        # If no image provided, we must have an existing mesh to post-process
        if not args.mesh:
            print("[Error] Must provide either --image, --sheet, or an existing --mesh.")
            sys.exit(1)
            
    # 2. Shape Reconstruction Stage (Skip if --mesh is specified)
    base_mesh_path = args.mesh
    
    if not base_mesh_path:
        print("\n--- STAGE 1: Lightweight Shape Generation (Hunyuan3D-Shape) ---")
        print("[shape] Loading Hunyuan3D Shape Pipeline...")
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2.1',
            device=args.device,
            torch_dtype=torch.float16,
        )
        
        print(f"[shape] Generating 3D shape from Front view...")
        print(f"        - Octree resolution: {args.octree_resolution}")
        print(f"        - Inference steps: {args.steps}")
        print(f"        - Guidance scale: {args.guidance_scale}")
        
        with torch.no_grad():
            mesh = shape_pipeline(
                image=front_image_path,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                octree_resolution=args.octree_resolution,
            )[0]
            
        base_mesh_path = os.path.join(args.output_dir, "base_geometry.glb")
        exported = mesh.export(base_mesh_path)
        
        # Fix trimesh export write issue
        if not os.path.exists(base_mesh_path):
            if isinstance(exported, bytes) and len(exported) > 0:
                with open(base_mesh_path, 'wb') as f:
                    f.write(exported)
                print("[shape] Saved raw GLB bytes successfully.")
            else:
                raise RuntimeError("[shape] Failed to export base geometry mesh.")
                
        print(f"[SUCCESS] Base architectural geometry saved: {base_mesh_path}")
        
        # Clean up heavy GPU structures immediately
        del mesh, shape_pipeline
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"\n[*] Skipping Stage 1: Reusing existing mesh {base_mesh_path}")

    # 3. Load & Process Geometry
    print("\n--- STAGE 2: Advanced Geometric Post-Processing ---")
    scene = trimesh.load(base_mesh_path)
    
    if isinstance(scene, trimesh.Scene):
        print("[*] Merging multi-geometry scene into a single mesh...")
        if len(scene.geometry) == 0:
            raise ValueError("No valid geometry found in loaded GLB scene.")
        mesh = scene.dump(concatenate=True)
        if isinstance(mesh, list):
            mesh = trimesh.util.concatenate(mesh)
    else:
        mesh = scene
        
    # Operator A: Ground Locking
    if args.lock_ground:
        mesh = apply_ground_locking(mesh, ratio=args.ground_ratio, target_z=args.ground_z)
        
    # Operator B: Smoothing
    mesh = apply_mesh_smoothing(
        mesh, 
        method=args.smoothing_method, 
        iterations=args.smoothing_iterations, 
        lamb=args.smoothing_lamb
    )
    
    # Save the processed mesh back to disk
    processed_mesh_path = os.path.join(args.output_dir, "processed_geometry.glb")
    mesh.export(processed_mesh_path)
    print(f"[+] Saved processed watertight mesh to: {processed_mesh_path}")
    
    # Operator C: Contour Slicing (SVG & DXF)
    if args.slicing_interval > 0.0:
        print("\n--- STAGE 3: Contour Slicing for Waffle Grid / Fabrication ---")
        generate_contour_slices(
            mesh, 
            interval=args.slicing_interval, 
            output_dir=args.output_dir, 
            base_name="architectural"
        )
        
    # 4. CAD & Fabrication-Ready Exports
    print("\n--- STAGE 4: Automated Engineering CAD Exporting ---")
    convert_glb_to_cad(
        glb_path=processed_mesh_path,
        output_dir=args.output_dir,
        scale_factor=args.scale,
        target_unit=args.unit,
        decimate_fraction=args.decimate,
        sample_points=args.points
    )
    
    print("\n=====================================================================")
    print("[SUCCESS] Architectural Fabrication pipeline finished successfully!")
    print(f"    Outputs saved in: {args.output_dir}")
    print("=====================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Specialized Architectural Shape Generation & CAD/Fabrication Pipeline"
    )
    # ── Input Configuration ──────────────────────────────────────────────────
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single 2D concept sketch (uses directly, no slicing).")
    parser.add_argument("--sheet", type=str, default=None,
                        help="Path to a horizontal multi-view turnaround sheet (will slice).")
    parser.add_argument("--num_views", type=int, default=4,
                        help="Number of views to slice from --sheet (default: 4).")
    parser.add_argument("--mesh", type=str, default=None,
                        help="Path to an existing mesh GLB/OBJ (skips Stage 1 shape generation).")
    
    # ── Slicing Configuration ───────────────────────────────────────────────
    parser.add_argument("--use_rembg", action="store_true",
                        help="Use AI background removal (rembg) when slicing sheet.")
    parser.add_argument("--bg_threshold", type=int, default=240,
                        help="White threshold for sheet slicing if --use_rembg is disabled.")
                        
    # ── Shape Detail Parameters ─────────────────────────────────────────────
    parser.add_argument("--octree_resolution", type=int, default=512,
                        help="Marching cubes grid resolution (default: 512, range 256-640).")
    parser.add_argument("--steps", type=int, default=50,
                        help="Diffusion steps (default: 50).")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Classifier-free guidance scale (default: 5.0).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device to use (default: cuda:0).")
                        
    # ── Ground Locking Parameters ───────────────────────────────────────────
    parser.add_argument("--lock_ground", action="store_true", default=True,
                        help="Flatten lowest vertices and snap bottom to ground level.")
    parser.add_argument("--ground_ratio", type=float, default=0.05,
                        help="Fraction of bottom bounding-box height to flatten (default: 0.05).")
    parser.add_argument("--ground_z", type=float, default=0.0,
                        help="Ground plane Z elevation (default: 0.0).")
                        
    # ── Mesh Smoothing Parameters ───────────────────────────────────────────
    parser.add_argument("--smoothing_method", type=str, default="laplacian",
                        choices=["laplacian", "taubin", "none"],
                        help="Mesh smoothing filter algorithm (default: laplacian).")
    parser.add_argument("--smoothing_iterations", type=int, default=10,
                        help="Number of smoothing passes (default: 10).")
    parser.add_argument("--smoothing_lamb", type=float, default=0.5,
                        help="Smoothing step factor (default: 0.5).")
                        
    # ── Slicing & Contour Parameters ─────────────────────────────────────────
    parser.add_argument("--slicing_interval", type=float, default=0.2,
                        help="Contour vertical slice interval in meters/units (default: 0.2, 0 to disable).")
                        
    # ── Engineering Export Options (glb_to_cad.py) ───────────────────────────
    parser.add_argument("--scale", default="auto",
                        help="Metric normalization factor. Use 'auto' (default) or float like '0.01'.")
    parser.add_argument("--unit", default="mm", choices=["m", "cm", "mm"],
                        help="Physical STL output unit (m, cm, or mm. Default: mm).")
    parser.add_argument("--decimate", type=float, default=0.0,
                        help="Adaptive face decimation ratio (0.0 to 1.0, default: 0.0).")
    parser.add_argument("--points", type=int, default=100000,
                        help="Point density for reference colored PLY cloud (default: 100000).")
    parser.add_argument("--output_dir", type=str, default="./output_architectural",
                        help="Directory to save all final assets.")
                        
    args = parser.parse_args()
    
    # Validation
    if not args.image and not args.sheet and not args.mesh:
        parser.print_help()
        sys.exit(1)
        
    run_architectural_pipeline(args)
