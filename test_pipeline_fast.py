#!/usr/bin/env python3
"""
Fast geometry pipeline test — no ML model loading required.
Tests the parts that have been failing: GLB→OBJ, remesh, UV wrap,
meshVerticeInpaint patch, OpenCV inpaint fallback, GLB export.
Expected runtime: ~30 seconds.
"""
import os, sys, traceback
from unittest.mock import MagicMock

try:
    import bpy  # noqa: F401
except ImportError:
    sys.modules['bpy'] = MagicMock()

import torchvision.transforms.functional as _tvf
if 'torchvision.transforms.functional_tensor' not in sys.modules:
    sys.modules['torchvision.transforms.functional_tensor'] = _tvf

WORKSPACE = '/kaggle/working'
sys.path.insert(0, os.path.join(WORKSPACE, 'Hunyuan3D-2.1', 'hy3dpaint'))
sys.path.insert(0, WORKSPACE)

try:
    import custom_rasterizer
    if not hasattr(custom_rasterizer, 'rasterize'):
        import custom_rasterizer.custom_rasterizer as _cr
        sys.modules['custom_rasterizer'] = _cr
except ImportError:
    pass

import numpy as np
import trimesh

RESULTS = {}
TMP = '/tmp/pipeline_test'
os.makedirs(TMP, exist_ok=True)


def run(name, fn, *args):
    """Run a test, record result, return value or None on failure."""
    try:
        val = fn(*args)
        RESULTS[name] = 'PASS'
        print(f"  [PASS] {name}")
        return val
    except Exception as e:
        RESULTS[name] = f'FAIL: {e}'
        print(f"  [FAIL] {name}")
        traceback.print_exc()
        return None


# ── 1. Create test sphere GLB ─────────────────────────────────────────────────
def make_glb():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    path = os.path.join(TMP, 'sphere.glb')
    sphere.export(path)
    assert os.path.exists(path)
    return path


# ── 2. GLB → OBJ ─────────────────────────────────────────────────────────────
def glb_to_obj(glb_path):
    from multiview_utils.multiview_paint_pipeline_v2 import _ensure_obj
    obj = _ensure_obj(glb_path, TMP)
    assert os.path.exists(obj)
    return obj


# ── 3. remesh_mesh ────────────────────────────────────────────────────────────
def do_remesh(obj_path):
    from utils.simplify_mesh_utils import remesh_mesh
    out = os.path.join(TMP, 'remesh.obj')
    remesh_mesh(obj_path, out)
    assert os.path.exists(out)
    return out


# ── 4. mesh_uv_wrap ───────────────────────────────────────────────────────────
def do_uv_wrap(obj_path):
    from utils.uvwrap_utils import mesh_uv_wrap
    mesh = trimesh.load(obj_path)
    mesh = mesh_uv_wrap(mesh)
    return mesh


# ── 5. OpenCV TELEA fallback (direct) ─────────────────────────────────────────
def do_opencv_inpaint():
    from multiview_utils.multiview_paint_pipeline_v2 import _opencv_inpaint_fallback
    tex = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    result, result_mask = _opencv_inpaint_fallback(tex, mask, None, None, None, None)
    assert result.shape == tex.shape
    assert result_mask.sum() == 0


# ── 6. MeshRender patch ───────────────────────────────────────────────────────
def do_meshrender_patch():
    from multiview_utils.multiview_paint_pipeline_v2 import _patch_mesh_render_if_needed
    _patch_mesh_render_if_needed()
    mr = sys.modules.get('DifferentiableRenderer.MeshRender')
    if mr is not None:
        assert hasattr(mr, 'meshVerticeInpaint')


# ── 7. Trimesh GLB export ─────────────────────────────────────────────────────
def do_glb_export(mesh):
    out = os.path.join(TMP, 'output.glb')
    mesh.export(out)
    assert os.path.getsize(out) > 0
    return out


# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("[*] Hunyuan3D v2 — Fast Geometry Pipeline Test")
print("[*] No ML models | Expected runtime: ~30s")
print("=" * 60)

glb  = run("1. Create sphere GLB",  make_glb)
obj  = run("2. GLB → OBJ",         glb_to_obj,   glb)  if glb  else None
robj = run("3. remesh_mesh",        do_remesh,    obj)  if obj  else None
mesh = run("4. mesh_uv_wrap",       do_uv_wrap,   robj) if robj else None
run("5. OpenCV TELEA inpaint",      do_opencv_inpaint)
run("6. MeshRender patch",          do_meshrender_patch)
if mesh:
    run("7. GLB export",            do_glb_export, mesh)

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for v in RESULTS.values() if v == 'PASS')
total  = len(RESULTS)
print("\n" + "=" * 60)
print(f"Results: {passed}/{total} passed")
if passed < total:
    for name, status in RESULTS.items():
        if status != 'PASS':
            print(f"  ✗ {name}: {status}")
    sys.exit(1)
else:
    print("[SUCCESS] All geometry pipeline components OK")
print("=" * 60)
