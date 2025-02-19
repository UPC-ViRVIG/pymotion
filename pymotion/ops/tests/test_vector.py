import pymotion.ops.vector as vec
import pymotion.ops.vector_torch as vec_torch
import numpy as np
import torch
from numpy.testing import assert_allclose


class TestVector:
    atol = 1e-6
    low_atol = 1e-3  # for those operations that are not as precise

    def test_normalize_numpy(self):
        # Test with zero vector
        v_zero_np = np.array([0.0, 0.0, 0.0])
        normalized_v_zero_np = vec.normalize(v_zero_np)
        assert_allclose(
            normalized_v_zero_np,
            np.array([0.0, 0.0, 0.0]),
            atol=self.atol,
            err_msg="Zero vector NumPy failed",
        )

        # Test with unit vector
        v_unit_np = np.array([1.0, 0.0, 0.0])
        normalized_v_unit_np = vec.normalize(v_unit_np)
        assert_allclose(normalized_v_unit_np, v_unit_np, atol=self.atol, err_msg="Unit vector NumPy failed")

        # Test with random vectors
        np.random.seed(1)
        v_rand_np = np.random.rand(10, 3)
        normalized_v_rand_np = vec.normalize(v_rand_np)
        for i in range(10):
            norm = np.linalg.norm(normalized_v_rand_np[i])
            assert_allclose(norm, 1.0, atol=self.atol, err_msg=f"Random vector NumPy failed for index {i}")

        # Test with vectors of different magnitudes
        v_mag_np = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.5, 1.0, 1.5]])
        normalized_v_mag_np = vec.normalize(v_mag_np)
        for i in range(3):
            norm = np.linalg.norm(normalized_v_mag_np[i])
            assert_allclose(norm, 1.0, atol=self.atol, err_msg=f"Magnitude vector NumPy failed for index {i}")

    def test_normalize_torch(self):
        # Test with zero vector
        v_zero_torch = torch.tensor([0.0, 0.0, 0.0])
        normalized_v_zero_torch = vec_torch.normalize(v_zero_torch)
        assert_allclose(
            normalized_v_zero_torch.numpy(),
            np.array([0.0, 0.0, 0.0]),
            atol=self.atol,
            err_msg="Zero vector PyTorch failed",
        )

        # Test with unit vector
        v_unit_torch = torch.tensor([1.0, 0.0, 0.0])
        normalized_v_unit_torch = vec_torch.normalize(v_unit_torch)
        assert_allclose(
            normalized_v_unit_torch.numpy(),
            v_unit_torch.numpy(),
            atol=self.atol,
            err_msg="Unit vector PyTorch failed",
        )

        # Test with random vectors
        torch.manual_seed(1)
        v_rand_torch = torch.rand(10, 3)
        normalized_v_rand_torch = vec_torch.normalize(v_rand_torch)
        for i in range(10):
            norm = torch.linalg.norm(normalized_v_rand_torch[i])
            assert_allclose(
                norm.numpy(), 1.0, atol=self.atol, err_msg=f"Random vector PyTorch failed for index {i}"
            )

        # Test with vectors of different magnitudes
        v_mag_torch = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.5, 1.0, 1.5]])
        normalized_v_mag_torch = vec_torch.normalize(v_mag_torch)
        for i in range(3):
            norm = torch.linalg.norm(normalized_v_mag_torch[i])
            assert_allclose(
                norm.numpy(), 1.0, atol=self.atol, err_msg=f"Magnitude vector PyTorch failed for index {i}"
            )
