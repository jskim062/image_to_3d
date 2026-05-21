import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw
from unittest.mock import MagicMock, patch

# Add workspace directory to python path
workspace_dir = os.path.abspath(os.path.dirname(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

# Mock missing heavy GPU/C++ and library dependencies for the CPU test environment
import torch
from unittest.mock import MagicMock

# Define dummy base classes for Hunyuan3D pipeline
class DummyPaintPipeline:
    def __init__(self, config=None):
        self.config = config if config is not None else DummyPaintConfig(6, 512)
        self.render = MagicMock()
        
        self.view_processor = MagicMock()
        self.view_processor.bake_view_selection.return_value = (
            [0, 0, 0, 0, 90, -90], 
            [0, 90, 180, 270, 0, 180], 
            [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        )
        self.view_processor.render_normal_multiview.return_value = [MagicMock()] * 6
        self.view_processor.render_position_multiview.return_value = [MagicMock()] * 6
        self.view_processor.bake_from_multiview.return_value = (
            MagicMock(), 
            torch.zeros((1, 1, 1))
        )
        
        # Mock models dictionary
        mock_multiview = MagicMock()
        mock_multiview.return_value = {
            "albedo": [Image.new("RGB", (64, 64)) for _ in range(6)],
            "mr": [Image.new("RGB", (64, 64)) for _ in range(6)]
        }
        self.models = {
            "multiview_model": mock_multiview,
            "super_model": MagicMock(side_effect=lambda x: x)
        }

class DummyPaintConfig:
    def __init__(self, max_num_view, resolution):
        self.render_size = 512
        self.resolution = resolution
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        self.max_selected_view_num = max_num_view
        self.realesrgan_ckpt_path = ""
        self.multiview_cfg_path = ""

# Create global mock objects to trace function calls
mock_mesh = MagicMock()
mock_shape_inst = MagicMock()
mock_shape_inst.return_value = [mock_mesh]
mock_shape_pipeline_class = MagicMock()
mock_shape_pipeline_class.from_pretrained.return_value = mock_shape_inst

mock_paint_inst = MagicMock()
mock_paint_pipeline_class = MagicMock()
mock_paint_pipeline_class.return_value = mock_paint_inst

# Inject into sys.modules to intercept imports completely
sys.modules["pymeshlab"] = MagicMock()
sys.modules["custom_rasterizer"] = MagicMock()
sys.modules["mesh_inpaint_processor"] = MagicMock()
sys.modules["utils.uvwrap_utils"] = MagicMock()
sys.modules["utils.simplify_mesh_utils"] = MagicMock()
sys.modules["DifferentiableRenderer.mesh_utils"] = MagicMock()

# Inject textureGenPipeline
mock_texture_pipeline = MagicMock()
mock_texture_pipeline.Hunyuan3DPaintPipeline = DummyPaintPipeline
mock_texture_pipeline.Hunyuan3DPaintConfig = DummyPaintConfig
sys.modules["textureGenPipeline"] = mock_texture_pipeline

# Inject shape pipeline
sys.modules["hy3dshape.pipelines"] = MagicMock(Hunyuan3DDiTFlowMatchingPipeline=mock_shape_pipeline_class)

# Inject custom paint pipeline
sys.modules["multiview_utils.multiview_paint_pipeline"] = MagicMock(MultiViewPaintPipeline=mock_paint_pipeline_class)

# Inject diffusers and transformers to prevent version conflicts on CPU
sys.modules["diffusers"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["accelerate"] = MagicMock()

from multiview_utils.image_utils import slice_turnaround_sheet, remove_background_and_center, align_multiview_colors
from multiview_pipeline import run_multiview_pipeline

class TestMultiViewSystem(unittest.TestCase):
    
    def setUp(self):
        self.output_dir = os.path.join(workspace_dir, "test_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Reset mocks
        mock_shape_pipeline_class.reset_mock()
        mock_paint_pipeline_class.reset_mock()
        mock_shape_inst.reset_mock()
        mock_mesh.reset_mock()
        mock_paint_inst.reset_mock()
        
        # Create a mock 4-view horizontal sheet
        # Each view is 100x100 (total 400x100)
        self.sheet_4view_path = os.path.join(self.output_dir, "dummy_4view_sheet.png")
        self.create_dummy_sheet(self.sheet_4view_path, num_views=4)
        
        # Create a mock 6-view horizontal sheet
        # Each view is 100x100 (total 600x100)
        self.sheet_6view_path = os.path.join(self.output_dir, "dummy_6view_sheet.png")
        self.create_dummy_sheet(self.sheet_6view_path, num_views=6)
        
        # Create a dummy base mesh path
        self.dummy_mesh_path = os.path.join(self.output_dir, "dummy_mesh.obj")
        with open(self.dummy_mesh_path, "w") as f:
            f.write("# Dummy OBJ File\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    def tearDown(self):
        # Clean up files
        for filename in ["dummy_4view_sheet.png", "dummy_6view_sheet.png", "dummy_mesh.obj", "final_multiview_result.obj", "final_multiview_result.glb", "base_geometry.glb"]:
            path = os.path.join(self.output_dir, filename)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        for i in range(6):
            path = os.path.join(self.output_dir, f"sliced_view_{i}.png")
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        if os.path.exists(self.output_dir):
            try:
                os.rmdir(self.output_dir)
            except Exception:
                pass

    def create_dummy_sheet(self, path, num_views=4):
        # Create image with solid white background
        width = num_views * 100
        height = 100
        sheet = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        
        # Colors for each segment to distinguish them
        colors = [
            (255, 0, 0),    # Red (Front)
            (0, 255, 0),    # Green (Left/Right)
            (0, 0, 255),    # Blue (Back)
            (255, 255, 0),  # Yellow (Left/Right)
            (255, 0, 255),  # Magenta (Top)
            (0, 255, 255)   # Cyan (Bottom)
        ]
        
        for i in range(num_views):
            left = i * 100
            right = (i + 1) * 100
            # Draw a colored square in the center of the segment (isolated from edges)
            draw.rectangle([left + 20, 20, right - 20, 80], fill=colors[i % len(colors)])
            
        sheet.save(path)

    def test_image_slicing_auto_detect_4view(self):
        """Tests that a 4-view horizontal sheet is auto-detected and correctly sliced."""
        views = slice_turnaround_sheet(self.sheet_4view_path)
        self.assertEqual(len(views), 4)
        for view in views:
            self.assertEqual(view.size, (512, 512))
            self.assertEqual(view.mode, "RGB")

    def test_image_slicing_auto_detect_6view(self):
        """Tests that a 6-view horizontal sheet is auto-detected and correctly sliced."""
        views = slice_turnaround_sheet(self.sheet_6view_path)
        self.assertEqual(len(views), 6)
        for view in views:
            self.assertEqual(view.size, (512, 512))
            self.assertEqual(view.mode, "RGB")

    def test_remove_background_and_centering(self):
        """Tests background removal and centering on standard images."""
        # Create an image with an off-center object
        img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 30, 40, 70], fill=(255, 0, 0, 255)) # Off-center red box
        
        processed = remove_background_and_center(img, bg_threshold=240, target_size=(256, 256))
        self.assertEqual(processed.size, (256, 256))
        
        # Verify the red box is now centered on the white background
        data = np.array(processed)
        r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        
        # Non-white mask
        mask = (r < 240) | (g < 240) | (b < 240)
        coords = np.argwhere(mask)
        self.assertTrue(len(coords) > 0)
        
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        center_y = (y0 + y1) // 2
        center_x = (x0 + x1) // 2
        
        # The center of the red box should be close to the image center (128, 128)
        self.assertAlmostEqual(center_y, 128, delta=5)
        self.assertAlmostEqual(center_x, 128, delta=5)

    def test_align_multiview_colors(self):
        """Tests that color alignment (histogram matching) runs without error."""
        # Front image: Reddish tone
        front = Image.new("RGB", (100, 100), (200, 50, 50))
        # Side image: Greenish tone
        side = Image.new("RGB", (100, 100), (50, 180, 50))
        
        aligned_views = align_multiview_colors([front, side])
        self.assertEqual(len(aligned_views), 2)
        self.assertEqual(aligned_views[0], front)
        
        # Side view should now be shifted toward the reddish profile of front
        side_aligned_arr = np.array(aligned_views[1])
        r_avg = side_aligned_arr[:, :, 0].mean()
        g_avg = side_aligned_arr[:, :, 1].mean()
        self.assertTrue(r_avg > g_avg, f"Histogram alignment failed. Red channel: {r_avg:.2f}, Green channel: {g_avg:.2f}")

    def test_pipeline_dry_run_reordering_4view(self):
        """
        Tests the orchestrator pipeline with a 4-view sheet.
        Verifies that view reordering works correctly and the sub-pipelines are called with correct parameters.
        """
        # Configure Mock Shape Pipeline
        mock_shape_inst = MagicMock()
        mock_mesh = MagicMock()
        mock_shape_inst.return_value = [mock_mesh]
        mock_shape_pipeline_class.from_pretrained.return_value = mock_shape_inst
        
        # Configure Mock Paint Pipeline
        mock_paint_inst = MagicMock()
        mock_paint_inst.return_value = os.path.join(self.output_dir, "final_multiview_result.glb")
        mock_paint_pipeline_class.return_value = mock_paint_inst
        
        # Set up pipeline CLI arguments
        class Args:
            sheet = self.sheet_4view_path
            num_views = 4
            view_order = "front_left_back_right"
            mesh = None
            resolution = 256
            bg_threshold = 240
            device = "cpu"
            output_dir = self.output_dir
            
        args = Args()
        
        # Run orchestrator
        run_multiview_pipeline(args)
        
        # Verify shape pipeline was loaded and called
        mock_shape_pipeline_class.from_pretrained.assert_called_once_with(
            'tencent/Hunyuan3D-2.1',
            device="cpu",
            torch_dtype=unittest.mock.ANY,
        )
        mock_mesh.export.assert_called_once_with(os.path.join(self.output_dir, "base_geometry.glb"))
        
        # Verify paint pipeline was instantiated and called
        mock_paint_pipeline_class.assert_called_once()
        mock_paint_inst.assert_called_once()
        
        # Retrieve keyword arguments passed to the mock paint pipeline
        call_kwargs = mock_paint_inst.call_args[1]
        self.assertEqual(call_kwargs["mesh_path"], os.path.join(self.output_dir, "base_geometry.glb"))
        self.assertTrue(call_kwargs["use_remesh"])
        self.assertTrue(call_kwargs["save_glb"])
        
        # Verify the views list was reordered to match camera requirements
        # Sliced order was: [Red, Green, Blue, Yellow] (Front, Left, Back, Right)
        # Expected reordered list: [Front, Right, Back, Left] = [Red, Yellow, Blue, Green]
        passed_views = call_kwargs["views"]
        self.assertEqual(len(passed_views), 4)
        
        # Check colors of first pixels to verify correct view assignment
        red_pixel = np.array(passed_views[0])[50, 50] # Front
        yellow_pixel = np.array(passed_views[1])[50, 50] # Right
        blue_pixel = np.array(passed_views[2])[50, 50] # Back
        green_pixel = np.array(passed_views[3])[50, 50] # Left
        
        self.assertEqual(red_pixel[0], 255) # Red (R=255, G=0, B=0)
        self.assertEqual(yellow_pixel[0], 255) # Yellow (R=255, G=255, B=0)
        self.assertEqual(blue_pixel[2], 255) # Blue (R=0, G=0, B=255)
        self.assertEqual(green_pixel[1], 255) # Green (R=0, G=255, B=0)

    def test_pipeline_dry_run_reordering_6view(self):
        """
        Tests the orchestrator pipeline with a 6-view sheet.
        Verifies that view reordering and Top/Bottom mappings work correctly.
        """
        # Configure Mock Shape Pipeline
        mock_shape_inst = MagicMock()
        mock_mesh = MagicMock()
        mock_shape_inst.return_value = [mock_mesh]
        mock_shape_pipeline_class.from_pretrained.return_value = mock_shape_inst
        
        # Configure Mock Paint Pipeline
        mock_paint_inst = MagicMock()
        mock_paint_inst.return_value = os.path.join(self.output_dir, "final_multiview_result.glb")
        mock_paint_pipeline_class.return_value = mock_paint_inst
        
        # Set up pipeline CLI arguments with an existing mesh to skip shape generation
        class Args:
            sheet = self.sheet_6view_path
            num_views = 6
            view_order = "front_left_back_right"
            mesh = self.dummy_mesh_path
            resolution = 256
            bg_threshold = 240
            device = "cpu"
            output_dir = self.output_dir
            
        args = Args()
        
        # Run orchestrator
        run_multiview_pipeline(args)
        
        # Verify shape pipeline was NOT called because mesh was provided
        mock_shape_pipeline_class.from_pretrained.assert_not_called()
        
        # Retrieve keyword arguments passed to the mock paint pipeline
        call_kwargs = mock_paint_inst.call_args[1]
        self.assertEqual(call_kwargs["mesh_path"], self.dummy_mesh_path)
        
        # Verify the views list was reordered to match camera requirements
        # Sliced order was: [Red, Green, Blue, Yellow, Magenta, Cyan] (Front, Left, Back, Right, Top, Bottom)
        # Expected reordered list: [Front, Right, Back, Left, Top, Bottom] = [Red, Yellow, Blue, Green, Magenta, Cyan]
        passed_views = call_kwargs["views"]
        self.assertEqual(len(passed_views), 6)
        
        # Check colors of first pixels to verify correct view assignment
        red_pixel = np.array(passed_views[0])[50, 50] # Front
        yellow_pixel = np.array(passed_views[1])[50, 50] # Right
        blue_pixel = np.array(passed_views[2])[50, 50] # Back
        green_pixel = np.array(passed_views[3])[50, 50] # Left
        magenta_pixel = np.array(passed_views[4])[50, 50] # Top
        cyan_pixel = np.array(passed_views[5])[50, 50] # Bottom
        
        self.assertEqual(red_pixel[0], 255) # Red
        self.assertEqual(yellow_pixel[0], 255) # Yellow
        self.assertEqual(blue_pixel[2], 255) # Blue
        self.assertEqual(green_pixel[1], 255) # Green
        self.assertEqual(magenta_pixel[0], 255) # Magenta (R=255, G=0, B=255)
        self.assertEqual(cyan_pixel[1], 255) # Cyan (R=0, G=255, B=255)

    def test_cad_exporter_pipeline(self):
        """Tests that glb_to_cad correctly loads, repairs, and exports CAD formats."""
        from glb_to_cad import convert_glb_to_cad
        
        # Run conversion on our dummy mesh
        success = convert_glb_to_cad(
            glb_path=self.dummy_mesh_path,
            output_dir=self.output_dir,
            scale_factor=1.0,
            target_unit="mm",
            decimate_fraction=0.0,
            sample_points=50
        )
        
        self.assertTrue(success)
        
        # Check that outputs exist
        obj_out = os.path.join(self.output_dir, "dummy_mesh_quad_ready.obj")
        stl_out = os.path.join(self.output_dir, "dummy_mesh_fabrication.stl")
        ply_out = os.path.join(self.output_dir, "dummy_mesh_pointcloud.ply")
        
        self.assertTrue(os.path.exists(obj_out), "Quad-ready OBJ was not created!")
        self.assertTrue(os.path.exists(stl_out), "Fabrication STL was not created!")
        self.assertTrue(os.path.exists(ply_out), "Pointcloud PLY was not created!")
        
        # Clean up CAD output files
        for p in [obj_out, stl_out, ply_out]:
            if os.path.exists(p):
                os.remove(p)

if __name__ == "__main__":
    unittest.main()

