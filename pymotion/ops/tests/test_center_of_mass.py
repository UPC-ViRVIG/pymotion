import numpy as np
from pymotion.ops.center_of_mass import center_of_mass, human_center_of_mass
from pymotion.ops.center_of_mass_torch import (
    center_of_mass as center_of_mass_torch,
    human_center_of_mass as human_center_of_mass_torch,
)
from numpy.testing import assert_allclose
import torch


class TestCoM:
    atol = 1e-6
    low_atol = 1e-3  # for those operations that are not as precise

    def test_com(self):
        assert_allclose(
            center_of_mass(
                np.array([[[0, 0, 0]]]),
                np.array([[1]]),
            ),
            np.array([[0, 0, 0]]),
            atol=self.atol,
        )
        assert_allclose(
            center_of_mass(
                np.array([[[0, 0, 0], [1, 0, 0]]]),
                np.array([[0.5, 0.5]]),
            ),
            np.array([[0.5, 0, 0]]),
            atol=self.atol,
        )
        assert_allclose(
            center_of_mass(
                np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
                np.array([[0.3, 0.3, 0.4]]),
            ),
            np.array([[4.3, 5.3, 6.3]]),
            atol=self.atol,
        )
        assert_allclose(
            center_of_mass_torch(
                torch.tensor([[[0, 0, 0]]]),
                torch.tensor([[1]]),
            ),
            torch.tensor([[0, 0, 0]]),
            atol=self.atol,
        )
        assert_allclose(
            center_of_mass_torch(
                torch.tensor([[[0, 0, 0], [1, 0, 0]]]),
                torch.tensor([[0.5, 0.5]]),
            ),
            torch.tensor([[0.5, 0, 0]]),
            atol=self.atol,
        )
        assert_allclose(
            center_of_mass_torch(
                torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
                torch.tensor([[0.3, 0.3, 0.4]]),
            ),
            torch.tensor([[4.3, 5.3, 6.3]]),
            atol=self.atol,
        )
        assert_allclose(
            human_center_of_mass(
                np.array([[[2, 1, 5]]]),
                np.array([[[1, 3, 4]]]),
                np.array([[[1, 1, 1]]]),
                np.array([[[0, 7, 6]]]),
                np.array([[[2, 8, 1]]]),
            ),
            np.array([[1.6, 3.05, 4.3]]),
        )
        assert_allclose(
            human_center_of_mass_torch(
                torch.tensor([[[2, 1, 5]]]),
                torch.tensor([[[1, 3, 4]]]),
                torch.tensor([[[1, 1, 1]]]),
                torch.tensor([[[0, 7, 6]]]),
                torch.tensor([[[2, 8, 1]]]),
            ),
            torch.tensor([[1.6, 3.05, 4.3]]),
        )
