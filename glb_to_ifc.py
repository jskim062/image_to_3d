import argparse
import sys
import os
import trimesh
import numpy as np
import ifcopenshell
from ifcopenshell.api import run

def auto_detect_scale(mesh):
    """
    Intelligently auto-detect the units of the 3D mesh based on its bounding box extents,
    and return the appropriate scale factor to convert the model to Meters (SI standard for IFC).
    """
    extents = mesh.extents  # [dx, dy, dz] in original units
    max_span = float(max(extents))
    
    # 3D generated meshes often range either:
    # 1. Near unit scale (e.g. max span 0.5 to 5.0) -> already in Meters
    # 2. In Centimeters (e.g. a chair of 100cm high) -> max span 30.0 to 200.0
    # 3. In Millimeters (e.g. a chair of 1000mm high) -> max span 300.0 to 2000.0
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
    print(f"[Auto-Scale] Applying scale factor: {scale:.4f} (converting to meters)")
    return scale

def hollow_mesh(mesh, thickness=0.1):
    """
    Hollows out a solid watertight mesh by creating an inward-offset inner shell 
    with inverted normals, and combining it with the outer shell.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        return mesh
    
    print(f"[*] Hollowing solid mesh with wall thickness: {thickness}m...")
    
    # 1. Create the inner shell by moving vertices inwards along their normals
    inner = mesh.copy()
    normals = inner.vertex_normals
    
    # Move vertices inwards
    inner.vertices = inner.vertices - (thickness * normals)
    
    # 2. Invert face orientation of the inner mesh
    inner.invert()
    
    # 3. Concatenate outer and inner meshes
    hollowed = trimesh.util.concatenate([mesh, inner])
    return hollowed

def convert_glb_to_ifc(
    glb_path,
    ifc_path,
    ifc_class="IfcBuildingElementProxy",
    project_name="3D to IFC Project",
    site_name="Default Site",
    building_name="Default Building",
    storey_name="Default Storey",
    element_name="Converted Mesh",
    scale_factor=None,  # None means 'auto'
    merge_meshes=True,
    hollow=False,
    thickness=0.1
):
    """
    Converts a GLB/glTF/OBJ model to a structured IFC4 file.
    """
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"Input GLB file not found: {glb_path}")

    print(f"[*] Loading 3D model from: {glb_path}...")
    scene = trimesh.load(glb_path)
    
    # Extract geometries from the loaded scene/mesh
    meshes_to_convert = []
    
    if isinstance(scene, trimesh.Scene):
        if len(scene.geometry) == 0:
            raise ValueError("No geometries found in the GLB scene.")
        
        if merge_meshes:
            print("[*] Merging multiple meshes in the GLB into a single unified mesh...")
            # scene.dump(concatenate=True) combines all geometries into a single Trimesh
            merged_mesh = scene.dump(concatenate=True)
            if isinstance(merged_mesh, list):
                # If dump returns a list of meshes, concatenate them manually
                merged_mesh = trimesh.util.concatenate(merged_mesh)
            meshes_to_convert.append((element_name, merged_mesh))
        else:
            print(f"[*] Processing {len(scene.geometry)} meshes separately...")
            for name, geom in scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    # trimesh scene.graph stores the transform matrices for each node
                    try:
                        transform = scene.graph.get(name)[0]
                    except Exception:
                        transform = np.identity(4)
                    
                    # Create a copy and apply the transform to get world/scene-space coordinates
                    geom_copy = geom.copy()
                    geom_copy.apply_transform(transform)
                    meshes_to_convert.append((name, geom_copy))
    elif isinstance(scene, trimesh.Trimesh):
        meshes_to_convert.append((element_name, scene))
    else:
        raise ValueError(f"Unsupported geometry type loaded: {type(scene)}")
    
    # 2. Compute Scaling Factor
    # If the scale factor is set to 'auto' or None, we compute it from the bounding box of the overall model
    if scale_factor is None or str(scale_factor).lower() == "auto":
        # Combine all meshes to get the global bounding box for scale detection
        if len(meshes_to_convert) == 1:
            detected_scale = auto_detect_scale(meshes_to_convert[0][1])
        else:
            combined_mesh = trimesh.util.concatenate([m[1] for m in meshes_to_convert])
            detected_scale = auto_detect_scale(combined_mesh)
        actual_scale = detected_scale
    else:
        try:
            actual_scale = float(scale_factor)
            print(f"[Manual-Scale] Using user-specified scale factor: {actual_scale:.6f}")
        except ValueError:
            print(f"[Warning] Invalid scale factor specified: '{scale_factor}'. Defaulting to auto-detection.")
            combined_mesh = trimesh.util.concatenate([m[1] for m in meshes_to_convert])
            actual_scale = auto_detect_scale(combined_mesh)

    # 3. Create a clean IFC file (IFC4 schema)
    print("[*] Creating new IFC4 model...")
    model = run("project.create_file", version="IFC4")
    
    # Create the standard BIM spatial structure (Project must exist before assigning units)
    project = run("root.create_entity", model, ifc_class="IfcProject", name=project_name)
    
    # Assign default SI units (meters for length, square meters for area, cubic meters for volume)
    length_unit = run("unit.add_si_unit", model, unit_type="LENGTHUNIT", prefix=None)
    area_unit = run("unit.add_si_unit", model, unit_type="AREAUNIT")
    volume_unit = run("unit.add_si_unit", model, unit_type="VOLUMEUNIT")
    run("unit.assign_unit", model, units=[length_unit, area_unit, volume_unit])
    
    # Create the rest of the spatial structure
    site = run("root.create_entity", model, ifc_class="IfcSite", name=site_name)
    building = run("root.create_entity", model, ifc_class="IfcBuilding", name=building_name)
    storey = run("root.create_entity", model, ifc_class="IfcBuildingStorey", name=storey_name)
    
    # Aggregate structural elements: Project contains Site, Site contains Building, Building contains Storey
    run("aggregate.assign_object", model, relating_object=project, products=[site])
    run("aggregate.assign_object", model, relating_object=site, products=[building])
    run("aggregate.assign_object", model, relating_object=building, products=[storey])
    
    # Add geometric representation contexts
    context = run("context.add_context", model, context_type="Model")
    body_context = run(
        "context.add_context", 
        model, 
        context_type="Model", 
        context_identifier="Body", 
        target_view="MODEL_VIEW", 
        parent=context
    )
    
    # 4. Generate IFC geometry representations
    for name, mesh in meshes_to_convert:
        num_verts = len(mesh.vertices)
        num_faces = len(mesh.faces)
        print(f"[*] Processing element '{name}': {num_verts} vertices, {num_faces} faces")
        
        if num_verts == 0 or num_faces == 0:
            print(f"[Warning] Skipping empty mesh '{name}'.")
            continue
            
        # Create a working copy of the mesh and scale it to meters
        scaled_mesh = mesh.copy()
        scaled_mesh.vertices = scaled_mesh.vertices * actual_scale
        
        # Apply hollowing if requested
        if hollow:
            scaled_mesh = hollow_mesh(scaled_mesh, thickness=thickness)
            num_verts = len(scaled_mesh.vertices)
            num_faces = len(scaled_mesh.faces)
            print(f"[*] Hollowing applied. New structure: {num_verts} vertices, {num_faces} faces")
            
        # Convert vertices and faces for IFC
        raw_verts = scaled_mesh.vertices.tolist()
        scaled_verts = [tuple(float(c) for c in v) for v in raw_verts]
        vertices_list = [scaled_verts]
        
        # Convert faces (0-based indices)
        raw_faces = scaled_mesh.faces.tolist()
        faces_list = [tuple(int(idx) for idx in f) for f in raw_faces]
        faces_list_wrapped = [faces_list]
        
        # Create IFC Product Entity of the specified class (e.g., IfcBuildingElementProxy)
        try:
            element = run("root.create_entity", model, ifc_class=ifc_class, name=name)
        except Exception as e:
            print(f"[Error] Failed to create IFC entity of class '{ifc_class}': {e}")
            print("Defaulting to 'IfcBuildingElementProxy'.")
            element = run("root.create_entity", model, ifc_class="IfcBuildingElementProxy", name=name)
        
        # Place the physical product inside our Building Storey
        run("spatial.assign_container", model, relating_structure=storey, products=[element])
        
        # Add mesh representation to the body context
        representation = run(
            "geometry.add_mesh_representation",
            model,
            context=body_context,
            vertices=vertices_list,
            faces=faces_list_wrapped
        )
        
        # Associate representation with the physical product
        run("geometry.assign_representation", model, product=element, representation=representation)
        
    # Write completed model to file
    print(f"[*] Writing IFC model to: {ifc_path}...")
    model.write(ifc_path)
    print("[+] Conversion completed successfully!")
    return actual_scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GLB/glTF 3D models into IFC4 BIM files.")
    parser.add_argument("glb_path", help="Path to the input GLB or glTF file.")
    parser.add_argument("ifc_path", help="Path to save the output IFC file.")
    parser.add_argument("--class", dest="ifc_class", default="IfcBuildingElementProxy", 
                        help="IFC entity class (e.g. IfcWall, IfcBuildingElementProxy, IfcFurniture). Default: IfcBuildingElementProxy.")
    parser.add_argument("--name", default="Converted Mesh", help="Name of the converted element in the IFC model.")
    parser.add_argument("--scale", default="auto", help="Scaling factor. E.g. '0.001' for mm->m, or 'auto' (default) to auto-detect.")
    parser.add_argument("--no-merge", action="store_true", help="Process sub-meshes as separate elements instead of merging.")
    parser.add_argument("--project", default="3D to IFC Project", help="Name of the IfcProject.")
    parser.add_argument("--site", default="Default Site", help="Name of the IfcSite.")
    parser.add_argument("--building", default="Default Building", help="Name of the IfcBuilding.")
    parser.add_argument("--storey", default="Default Storey", help="Name of the IfcBuildingStorey.")
    parser.add_argument("--hollow", action="store_true", help="Hollow out the solid mesh to leave only walls.")
    parser.add_argument("--thickness", type=float, default=0.1, help="Wall thickness in meters when hollowing is enabled. Default: 0.1.")

    args = parser.parse_args()
    
    try:
        convert_glb_to_ifc(
            glb_path=args.glb_path,
            ifc_path=args.ifc_path,
            ifc_class=args.ifc_class,
            project_name=args.project,
            site_name=args.site,
            building_name=args.building,
            storey_name=args.storey,
            element_name=args.name,
            scale_factor=args.scale,
            merge_meshes=not args.no_merge,
            hollow=args.hollow,
            thickness=args.thickness
        )
    except Exception as e:
        print(f"\n[Fatal Error] {e}", file=sys.stderr)
        sys.exit(1)
