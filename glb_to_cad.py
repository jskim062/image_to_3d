import argparse
import sys
import os
import trimesh
import numpy as np

def auto_detect_scale(mesh):
    """
    Intelligently auto-detect the units of the 3D mesh based on its bounding box extents,
    and return the appropriate scale factor to convert the model to Meters (SI standard).
    """
    extents = mesh.extents  # [dx, dy, dz] in original units
    max_span = float(max(extents))
    
    if max_span > 250.0:
        scale = 0.001
        unit_name = "Millimeters (mm)"
    elif max_span > 25.0:
        scale = 0.01
        unit_name = "Centimeters (cm)"
    else:
        scale = 1.0
        unit_name = "Meters (m)"
        
    print(f"[Auto-Scale] Mesh bounding box extents: {extents}")
    print(f"[Auto-Scale] Max span is {max_span:.3f}. Auto-detected unit: {unit_name}.")
    print(f"[Auto-Scale] Applying scale factor: {scale:.4f} to normalize to Meters.")
    return scale, unit_name

def clean_and_repair_mesh(mesh):
    """
    Applies geometry sanitization: merges vertices, repairs normals, 
    and removes degenerate/duplicate faces to ensure a clean topological structure.
    """
    print("[*] Repairing and cleaning mesh geometry...")
    initial_verts = len(mesh.vertices)
    initial_faces = len(mesh.faces)
    
    # trimesh.process(validate=True) cleans up vertices, degenerate faces, and duplicate faces safely
    try:
        mesh.process(validate=True)
    except Exception as e:
        print(f"[Warning] Unified mesh process failed: {e}. Attempting manual fallback...")
        try:
            mesh.merge_vertices()
        except Exception:
            pass
        try:
            mesh.fix_normals()
        except Exception:
            pass
    
    final_verts = len(mesh.vertices)
    final_faces = len(mesh.faces)
    print(f"[+] Geometry Sanitized: Vertices {initial_verts} -> {final_verts}, Faces {initial_faces} -> {final_faces}")
    return mesh

def decimate_mesh(mesh, decimate_fraction):
    """
    Simplifies the mesh using quadric error metrics to keep detail on high-curvature edges
    while simplifying flat areas.
    """
    if decimate_fraction <= 0.0 or decimate_fraction >= 1.0:
        return mesh
        
    target_faces = int(len(mesh.faces) * decimate_fraction)
    print(f"[*] Decimating mesh from {len(mesh.faces)} to target {target_faces} faces ({decimate_fraction*100:.1f}% remaining)...")
    
    try:
        # trimesh quadratic decimation simplifies the mesh robustly
        simplified = mesh.simplify_quadric_decimation(target_faces)
        print(f"[+] Decimation completed. New face count: {len(simplified.faces)}")
        return simplified
    except Exception as e:
        print(f"[Warning] Quadric decimation failed: {e}. Falling back to original mesh.")
        return mesh

def convert_glb_to_cad(
    glb_path,
    output_dir,
    scale_factor=None,       # None/auto or float
    target_unit="m",         # "m", "cm", "mm"
    decimate_fraction=0.0,   # 0.0 means no decimation
    sample_points=100000     # Number of points for the PLY pointcloud
):
    """
    Loads a GLB/glTF/OBJ model, cleans and repairs the geometry, auto-scales it, 
    and exports:
      1. [name]_quad_ready.obj : Clean optimized mesh for Rhino QuadRemesh / SubD
      2. [name]_fabrication.stl : Watertight STL scaled to target engineering units for 3D Printing
      3. [name]_pointcloud.ply : High-density point cloud with vertex colors for CAD references
    """
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"Input file not found: {glb_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(glb_path))[0]
    
    print(f"[*] Loading 3D model: {glb_path}...")
    scene = trimesh.load(glb_path)
    
    # 1. Merge into a single Trimesh geometry if loaded as a Scene
    if isinstance(scene, trimesh.Scene):
        print("[*] Scene container detected. Merging sub-meshes...")
        if len(scene.geometry) == 0:
            raise ValueError("No geometry found in the scene.")
        mesh = scene.dump(concatenate=True)
        if isinstance(mesh, list):
            mesh = trimesh.util.concatenate(mesh)
    else:
        mesh = scene
        
    # 2. Extract texture maps to vertex colors before clearing materials for clean CAD import
    has_texture = False
    if hasattr(mesh.visual, 'to_color'):
        try:
            print("[*] Converting texture maps to high-fidelity vertex colors...")
            mesh.visual = mesh.visual.to_color()
            has_texture = True
        except Exception as e:
            print(f"[Warning] Texture-to-color conversion skipped: {e}")
            
    # 3. Unit Normalization to Meters
    if scale_factor is None or str(scale_factor).lower() == "auto":
        scale, orig_unit = auto_detect_scale(mesh)
    else:
        try:
            scale = float(scale_factor)
            orig_unit = "Manual"
            print(f"[Manual-Scale] Applying user-specified scale factor: {scale:.6f}")
        except ValueError:
            print(f"[Warning] Invalid scale factor '{scale_factor}'. Defaulting to auto-detection.")
            scale, orig_unit = auto_detect_scale(mesh)
            
    # Normalize vertices to Meters
    mesh.vertices = mesh.vertices * scale
    
    # 4. Mesh Sanitization & Repair
    mesh = clean_and_repair_mesh(mesh)
    
    # 5. Optional Decimation
    if decimate_fraction > 0.0:
        mesh = decimate_mesh(mesh, decimate_fraction)
        
    # 6. EXPORT 1: Rhino/SubD Quad-Friendly OBJ
    # We export the cleaned, normalized (and optionally decimated) mesh.
    obj_path = os.path.join(output_dir, f"{base_name}_quad_ready.obj")
    print(f"[*] Exporting CAD-ready quad-friendly mesh to: {obj_path}...")
    mesh.export(obj_path)
    
    # 7. EXPORT 2: Watertight scaled STL for Fabrication / 3D Printing
    # STL requires the desired physical print/manufacturing units (meters, cm, or mm).
    unit_scalers = {
        "m": 1.0,
        "cm": 100.0,
        "mm": 1000.0
    }
    unit_names = {
        "m": "Meters (m)",
        "cm": "Centimeters (cm)",
        "mm": "Millimeters (mm)"
    }
    
    target_scale = unit_scalers.get(target_unit.lower(), 1.0)
    target_name = unit_names.get(target_unit.lower(), "Meters (m)")
    
    fab_mesh = mesh.copy()
    if target_scale != 1.0:
        print(f"[*] Scaling STL model by {target_scale} to target unit: {target_name}...")
        fab_mesh.vertices = fab_mesh.vertices * target_scale
        
    stl_path = os.path.join(output_dir, f"{base_name}_fabrication.stl")
    print(f"[*] Exporting fabrication-ready STL to: {stl_path}...")
    fab_mesh.export(stl_path)
    
    # 8. EXPORT 3: High-density Point Cloud PLY with vertex colors
    ply_path = os.path.join(output_dir, f"{base_name}_pointcloud.ply")
    print(f"[*] Generating high-density colored point cloud ({sample_points} points)...")
    
    try:
        # Sample points on the surface
        sampled_points, face_indices = trimesh.sample.sample_surface(mesh, sample_points)
        
        # Extract face colors for the sampled points
        face_colors = mesh.visual.face_colors
        sampled_colors = face_colors[face_indices]
        
        # Create PointCloud
        pc = trimesh.points.PointCloud(vertices=sampled_points, colors=sampled_colors)
        print(f"[*] Exporting visual reference point cloud to: {ply_path}...")
        pc.export(ply_path)
    except Exception as e:
        print(f"[Warning] Surface sampling failed: {e}. Falling back to mesh vertices point cloud.")
        try:
            pc = trimesh.points.PointCloud(vertices=mesh.vertices, colors=mesh.visual.vertex_colors)
            pc.export(ply_path)
        except Exception as e2:
            print(f"[Error] Point cloud generation failed: {e2}")

    print(f"[+] CAD and Fabrication export finished successfully in: {output_dir}")
    print(f"    - Quad-ready mesh: {os.path.basename(obj_path)}")
    print(f"    - Watertight STL ({target_name}): {os.path.basename(stl_path)}")
    print(f"    - Point Cloud reference: {os.path.basename(ply_path)}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GLB/OBJ meshes to CAD & Fabrication-friendly formats.")
    parser.add_argument("input_path", help="Path to the input GLB, glTF, or OBJ file.")
    parser.add_argument("output_dir", help="Directory where optimized outputs will be saved.")
    parser.add_argument("--scale", default="auto", help="Scaling factor to normalize original model to Meters. E.g. '0.01' or 'auto' (default).")
    parser.add_argument("--unit", default="mm", choices=["m", "cm", "mm"], help="Physical output unit for the STL fabrication file. Default: mm.")
    parser.add_argument("--decimate", type=float, default=0.0, help="Simplification factor (0.0 to 1.0). E.g. 0.5 to keep 50% of faces. Default: 0.0 (disabled).")
    parser.add_argument("--points", type=int, default=100000, help="Number of points to sample for the PLY point cloud reference. Default: 100000.")
    
    args = parser.parse_args()
    
    try:
        convert_glb_to_cad(
            glb_path=args.input_path,
            output_dir=args.output_dir,
            scale_factor=args.scale,
            target_unit=args.unit,
            decimate_fraction=args.decimate,
            sample_points=args.points
        )
    except Exception as e:
        print(f"\n[Fatal Error] {e}", file=sys.stderr)
        sys.exit(1)
