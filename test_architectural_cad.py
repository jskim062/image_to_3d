import os
import sys
import unittest
import numpy as np
import trimesh
from unittest.mock import MagicMock, patch

# Ensure current directory is in system path
workspace_dir = os.path.abspath(os.path.dirname(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

# Mock heavy modules to prevent loading issues in CPU test environment
sys.modules["pymeshlab"] = MagicMock()
sys.modules["custom_rasterizer"] = MagicMock()
sys.modules["diffusers"] = MagicMock()
sys.modules["hy3dshape"] = MagicMock()
sys.modules["hy3dshape.pipelines"] = MagicMock()

# Import the geometric operators from the pipeline
from architectural_cad_pipeline import (
    apply_ground_locking,
    apply_mesh_smoothing,
    generate_contour_slices,
    run_architectural_pipeline
)

class TestArchitecturalCADPipeline(unittest.TestCase):

    def setUp(self):
        # Create a test directory for temporary outputs
        self.test_output_dir = os.path.join(workspace_dir, "test_architectural_output")
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create a basic bumpy cylinder/box representing a rough 3D reconstruction mesh
        # Radius 1.0, Height 2.0 (Z from -1.0 to 1.0)
        self.base_mesh = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=16)
        
        # Add slight bumps to the base vertices to simulate AI reconstruction artifacts
        noise = np.random.normal(0, 0.05, self.base_mesh.vertices.shape)
        self.base_mesh.vertices += noise

    def tearDown(self):
        # Clean up temporary test files
        import shutil
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_ground_locking_mathematical_correction(self):
        """
        Verify that apply_ground_locking flattens the bottom vertices correctly
        and translates the mesh bottom to exactly Z = 0.0.
        """
        mesh = self.base_mesh.copy()
        original_verts = mesh.vertices.copy()
        
        # Apply ground locking (lowest 15% height)
        ratio = 0.15
        target_z = 0.0
        locked_mesh = apply_ground_locking(mesh, ratio=ratio, target_z=target_z)
        
        z_coords = locked_mesh.vertices[:, 2]
        z_min = float(np.min(z_coords))
        
        # 1. Bounding box bottom must lie exactly at target_z
        self.assertAlmostEqual(z_min, target_z, places=5)
        
        # 2. Assert that vertices in the bottom fraction are perfectly flat
        z_max = float(np.max(z_coords))
        z_height = z_max - z_min
        z_threshold = target_z + z_height * ratio
        
        # Check that original bottom vertices indeed now lie exactly on the flat Z = 0.0 plane
        orig_z = original_verts[:, 2]
        orig_z_min = float(np.min(orig_z))
        orig_z_max = float(np.max(orig_z))
        orig_threshold = orig_z_min + (orig_z_max - orig_z_min) * ratio
        
        # Find which indices should have been flattened
        flattened_indices = np.where(orig_z <= orig_threshold)[0]
        for idx in flattened_indices:
            self.assertAlmostEqual(locked_mesh.vertices[idx, 2], target_z, places=5)

    def test_mesh_smoothing_operators(self):
        """
        Verify that Laplacian and Taubin smoothing execute without errors
        and modify the vertices (i.e. smooths the noise).
        """
        # A. Laplacian Smoothing
        mesh_lap = self.base_mesh.copy()
        v_orig = mesh_lap.vertices.copy()
        smoothed_lap = apply_mesh_smoothing(mesh_lap, method="laplacian", iterations=5)
        # Assert vertices modified
        self.assertFalse(np.array_equal(smoothed_lap.vertices, v_orig))
        
        # B. Taubin Smoothing
        mesh_tau = self.base_mesh.copy()
        v_orig_tau = mesh_tau.vertices.copy()
        smoothed_tau = apply_mesh_smoothing(mesh_tau, method="taubin", iterations=5)
        # Assert vertices modified
        self.assertFalse(np.array_equal(smoothed_tau.vertices, v_orig_tau))

    def test_contour_slicing_dxf_svg(self):
        """
        Verify that generate_contour_slices successfully performs vertical intersections
        and exports DXF and SVG contour assets.
        """
        mesh = self.base_mesh.copy()
        # Ensure base is flat and aligned for clean slicing
        mesh = apply_ground_locking(mesh, ratio=0.1, target_z=0.0)
        
        # Slice mesh every 0.3 units (Height is ~2.0, should yield ~5-6 slices)
        interval = 0.3
        dxf_path = generate_contour_slices(
            mesh, 
            interval=interval, 
            output_dir=self.test_output_dir, 
            base_name="test_contours"
        )
        
        # Assert combined DXF file exists and is not empty
        self.assertIsNotNone(dxf_path)
        self.assertTrue(os.path.exists(dxf_path))
        self.assertGreater(os.path.getsize(dxf_path), 0)
        
        # Assert SVG directory created and contains slice files
        svg_dir = os.path.join(self.test_output_dir, "test_contours_svg")
        self.assertTrue(os.path.exists(svg_dir))
        svg_files = [f for f in os.listdir(svg_dir) if f.endswith(".svg")]
        self.assertGreater(len(svg_files), 0)
        
        print(f"[Test] Successfully generated {len(svg_files)} SVG contours.")

    def test_end_to_end_pipeline_flow_with_mocked_model(self):
        """
        Tests the end-to-end execution of run_architectural_pipeline
        by reusing an existing base mesh (simulating the mesh parameter)
        or mocking the reconstruction stage entirely.
        """
        # Save self.base_mesh as a mock input mesh
        base_glb_path = os.path.join(self.test_output_dir, "mock_input_base.glb")
        self.base_mesh.export(base_glb_path)
        
        # Create argument mock
        args = MagicMock()
        args.image = None
        args.sheet = None
        args.mesh = base_glb_path  # Skips stage 1 reconstruction, testing Stage 2, 3, 4 end-to-end
        args.lock_ground = True
        args.ground_ratio = 0.05
        args.ground_z = 0.0
        args.smoothing_method = "laplacian"
        args.smoothing_iterations = 5
        args.smoothing_lamb = 0.5
        args.slicing_interval = 0.4
        args.scale = "auto"
        args.unit = "mm"
        args.decimate = 0.0
        args.points = 1000
        args.output_dir = self.test_output_dir
        
        # Run pipeline
        run_architectural_pipeline(args)
        
        # Assert final CAD outputs are created successfully
        # 1. Processed watertight mesh
        processed_glb = os.path.join(self.test_output_dir, "processed_geometry.glb")
        self.assertTrue(os.path.exists(processed_glb))
        
        # 2. STL fabrication mesh (metric-scaled)
        stl_fab = os.path.join(self.test_output_dir, "processed_geometry_fabrication.stl")
        self.assertTrue(os.path.exists(stl_fab))
        
        # 3. Quad-ready OBJ mesh for Rhino
        obj_quad = os.path.join(self.test_output_dir, "processed_geometry_quad_ready.obj")
        self.assertTrue(os.path.exists(obj_quad))
        
        # 4. Point cloud visual PLY
        ply_pc = os.path.join(self.test_output_dir, "processed_geometry_pointcloud.ply")
        self.assertTrue(os.path.exists(ply_pc))
        
        # 5. Combined DXF contours
        dxf_contours = os.path.join(self.test_output_dir, "architectural_3d_contours.dxf")
        self.assertTrue(os.path.exists(dxf_contours))


if __name__ == "__main__":
    unittest.main()
