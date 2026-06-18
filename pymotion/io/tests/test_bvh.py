import os
import copy
import tempfile

import numpy as np
from numpy.testing import assert_allclose
from pymotion.io.bvh import BVH

# Path to the test BVH file shipped with the ops tests
TEST_BVH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "ops", "tests", "test.bvh"
)


def _load_test_bvh():
    """Helper to load a fresh copy of the test BVH."""
    bvh = BVH()
    bvh.load(TEST_BVH_PATH)
    return bvh


class TestBVH:
    atol = 1e-6

    # ------------------------------------------------------------------ merge
    def test_merge(self):
        bvh1 = _load_test_bvh()
        bvh2 = _load_test_bvh()
        n_frames_1 = bvh1.data["positions"].shape[0]
        n_frames_2 = bvh2.data["positions"].shape[0]
        bvh1.merge(bvh2)
        assert bvh1.data["positions"].shape[0] == n_frames_1 + n_frames_2
        assert bvh1.data["rotations"].shape[0] == n_frames_1 + n_frames_2

    def test_merge_mismatch_raises(self):
        bvh1 = _load_test_bvh()
        bvh2 = _load_test_bvh()
        bvh2.data["names"][0] = "BadName"
        try:
            bvh1.merge(bvh2)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    # -------------------------------------------------------------- resample
    def test_resample_same_fps(self):
        bvh = _load_test_bvh()
        original_frames = bvh.data["positions"].shape[0]
        original_ft = bvh.data["frame_time"]
        bvh.resample(target_frame_time=original_ft)
        assert bvh.data["positions"].shape[0] == original_frames

    def test_resample_half_fps(self):
        bvh = _load_test_bvh()
        original_frames = bvh.data["positions"].shape[0]
        original_ft = bvh.data["frame_time"]
        # Double the frame time = half the fps
        bvh.resample(target_frame_time=original_ft * 2)
        expected = original_frames // 2
        # Should be roughly half (within rounding)
        assert abs(bvh.data["positions"].shape[0] - expected) <= 1

    def test_resample_with_target_fps(self):
        bvh = _load_test_bvh()
        bvh.resample(target_fps=30)
        assert abs(bvh.data["frame_time"] - 1.0 / 30) < 1e-9

    def test_resample_both_raises(self):
        bvh = _load_test_bvh()
        try:
            bvh.resample(target_fps=30, target_frame_time=1.0 / 30)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_resample_neither_raises(self):
        bvh = _load_test_bvh()
        try:
            bvh.resample()
            assert False, "Expected ValueError"
        except ValueError:
            pass

    # ---------------------------------------------------------- keep_joints
    def test_keep_joints(self):
        bvh = _load_test_bvh()
        n_joints = len(bvh.data["names"])
        keep = [0, 1, 2]  # Root + first two children
        bvh.keep_joints(keep)
        assert len(bvh.data["names"]) <= len(keep)

    # ------------------------------------------------------- remove_joints_name
    def test_remove_joints_name(self):
        bvh = _load_test_bvh()
        original_names = list(bvh.data["names"])
        to_remove = ["Head"]
        bvh.remove_joints_name(to_remove)
        assert "Head" not in bvh.data["names"]
        assert len(bvh.data["names"]) == len(original_names) - 1

    # -------------------------------------------------------- keep_joints_name
    def test_keep_joints_name(self):
        bvh = _load_test_bvh()
        keep = ["Hips", "LeftHip", "RightHip"]
        bvh.keep_joints_name(keep)
        for name in bvh.data["names"]:
            assert name in keep

    # ------------------------------------------------------- remove_root_joint
    def test_remove_root_joint(self):
        bvh = _load_test_bvh()
        original_root = bvh.data["names"][0]
        bvh.remove_root_joint()
        assert bvh.data["names"][0] != original_root

    # ---------------------------------------------------------- slice_frames
    def test_slice_frames_start(self):
        bvh = _load_test_bvh()
        original_frames = bvh.data["positions"].shape[0]
        bvh.slice_frames(start=10)
        assert bvh.data["positions"].shape[0] == original_frames - 10
        assert bvh.data["rotations"].shape[0] == original_frames - 10

    def test_slice_frames_start_end(self):
        bvh = _load_test_bvh()
        bvh.slice_frames(start=5, end=15)
        assert bvh.data["positions"].shape[0] == 10
        assert bvh.data["rotations"].shape[0] == 10

    # ------------------------------------------------------ joint_intersection
    def test_joint_intersection(self):
        bvh1 = _load_test_bvh()
        bvh2 = _load_test_bvh()
        # Remove a joint from bvh2 so intersection is smaller
        bvh2.remove_joints_name(["Head"])
        bvh1.joint_intersection(bvh2)
        assert "Head" not in bvh1.data["names"]

    # ------------------------------------------------------------------ loop
    def test_loop(self):
        bvh = _load_test_bvh()
        original_frames = bvh.data["positions"].shape[0]
        bvh.loop(3)
        assert bvh.data["positions"].shape[0] == original_frames * 3
        assert bvh.data["rotations"].shape[0] == original_frames * 3

    # ----------------------------------------------------------------- shift
    def test_shift(self):
        bvh = _load_test_bvh()
        original_pos = bvh.data["positions"].copy()
        original_rot = bvh.data["rotations"].copy()
        bvh.shift(10)
        # First 10 frames should be the last 10 of the original
        assert_allclose(bvh.data["positions"][:10], original_pos[-10:], atol=self.atol)
        assert_allclose(bvh.data["rotations"][:10], original_rot[-10:], atol=self.atol)

    # ---------------------------------------------------------- rotate_root
    def test_rotate_root_identity(self):
        bvh = _load_test_bvh()
        bvh_ref = _load_test_bvh()
        original_pos = bvh.data["positions"].copy()
        bvh.rotate_root(0.0, axis="y")
        assert_allclose(bvh.data["positions"], original_pos, atol=1e-4)
        # Compare quaternions instead of Euler angles (avoids 360° wrapping)
        q_orig, _, _, _, _, _ = bvh_ref.get_data()
        q_new, _, _, _, _, _ = bvh.get_data()
        # Quaternions q and -q represent the same rotation
        dot = np.sum(q_orig * q_new, axis=-1)
        assert_allclose(np.abs(dot), 1.0, atol=1e-4)

    def test_rotate_root_360(self):
        bvh = _load_test_bvh()
        bvh_ref = _load_test_bvh()
        original_pos = bvh.data["positions"].copy()
        bvh.rotate_root(360.0, axis="y")
        assert_allclose(bvh.data["positions"], original_pos, atol=1e-3)
        # Compare quaternions (avoids Euler angle wrapping issues)
        q_orig, _, _, _, _, _ = bvh_ref.get_data()
        q_new, _, _, _, _, _ = bvh.get_data()
        dot = np.sum(q_orig * q_new, axis=-1)
        assert_allclose(np.abs(dot), 1.0, atol=1e-3)

    def test_rotate_root_axes(self):
        for ax in ["x", "y", "z"]:
            bvh = _load_test_bvh()
            bvh.rotate_root(90.0, axis=ax)
            # Just check it doesn't crash and shapes are preserved
            assert bvh.data["positions"].shape == _load_test_bvh().data["positions"].shape

    # ------------------------------------------------ _generate_mirror_mapping
    def test_generate_mirror_mapping(self):
        bvh = _load_test_bvh()
        mapping = bvh._generate_mirror_mapping()
        names = list(bvh.data["names"])
        # Check symmetry: mapping is its own inverse
        for i in range(len(mapping)):
            assert mapping[mapping[i]] == i, f"Mapping not symmetric at index {i}"
        # Check that Left/Right pairs are mapped correctly
        for i, name in enumerate(names):
            if "Left" in name:
                mirror_name = name.replace("Left", "Right")
                if mirror_name in names:
                    j = names.index(mirror_name)
                    assert mapping[i] == j, f"{name} should map to {mirror_name}"
                    assert mapping[j] == i, f"{mirror_name} should map to {name}"

    # ----------------------------------------------------------------- mirror
    def test_mirror(self):
        bvh = _load_test_bvh()
        # Use only a few frames for speed
        bvh.slice_frames(0, 10)
        original_shape = bvh.data["positions"].shape
        bvh.mirror()
        assert bvh.data["positions"].shape == original_shape

    # -------------------------------------------------------- copy_joint_order
    def test_copy_joint_order_same(self):
        bvh = _load_test_bvh()
        ref = _load_test_bvh()
        bvh.copy_joint_order(ref)
        assert list(bvh.data["names"]) == list(ref.data["names"])

    def test_copy_joint_order_mismatch_raises(self):
        bvh = _load_test_bvh()
        ref = _load_test_bvh()
        ref.remove_joints_name(["Head"])
        try:
            bvh.copy_joint_order(ref)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    # ---------------------------------------------------- save/load roundtrip
    def test_save_load_roundtrip(self):
        bvh = _load_test_bvh()
        bvh.slice_frames(0, 50)  # Use few frames for speed
        with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
            tmppath = f.name
        try:
            bvh.save(tmppath)
            bvh2 = BVH()
            bvh2.load(tmppath)
            assert list(bvh2.data["names"]) == list(bvh.data["names"])
            assert bvh2.data["positions"].shape == bvh.data["positions"].shape
            assert bvh2.data["rotations"].shape == bvh.data["rotations"].shape
            assert_allclose(bvh2.data["positions"], bvh.data["positions"], atol=1e-3)
        finally:
            os.unlink(tmppath)
