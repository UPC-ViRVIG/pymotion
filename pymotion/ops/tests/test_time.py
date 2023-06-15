import numpy as np
import torch
from pymotion.ops.time import interpolate_positions
from pymotion.ops.time_torch import interpolate_positions as interpolate_positions_torch
from numpy.testing import assert_allclose


class TestSkeleton:
    atol = 1e-6
    low_atol = 1e-3  # for those operations that are not as precise

    def test_interpolate(self):
        pos = np.array(
            [
                [
                    [0, 0, 0],
                    [1, 1, 0],
                    [2, 0, 0],
                    [8, 0, 1],
                    [20, 0, 0],
                ],
                [
                    [1, 1, 1],
                    [1, 1, 0],
                    [2, 0, 0],
                    [8, 0, 1],
                    [20, 0, 0],
                ],
            ]
        )[np.newaxis, np.newaxis, ...]
        x = np.array([0, 2, 3, 4, 5])
        new_x = np.array([0.5, 1.75, 2.25, 3.75, 4, 5, 6, 7, 8])
        gt = np.array(
            [
                [
                    [
                        [
                            [0.25, 0.25, 0.0],
                            [0.875, 0.875, 0.0],
                            [1.25, 0.75, 0.0],
                            [6.5, 0.0, 0.75],
                            [8.0, 0.0, 1.0],
                            [20.0, 0.0, 0.0],
                            [32.0, 0.0, -1.0],
                            [44.0, 0.0, -2.0],
                            [56.0, 0.0, -3.0],
                        ],
                        [
                            [1.0, 1.0, 0.75],
                            [1.0, 1.0, 0.125],
                            [1.25, 0.75, 0.0],
                            [6.5, 0.0, 0.75],
                            [8.0, 0.0, 1.0],
                            [20.0, 0.0, 0.0],
                            [32.0, 0.0, -1.0],
                            [44.0, 0.0, -2.0],
                            [56.0, 0.0, -3.0],
                        ],
                    ]
                ]
            ]
        )
        assert_allclose(
            interpolate_positions(new_x, x, pos, axis=3), gt, atol=self.atol
        )
        assert_allclose(
            interpolate_positions_torch(
                torch.from_numpy(new_x),
                torch.from_numpy(x),
                torch.from_numpy(pos),
                dim=3,
            ).numpy(),
            gt,
        )
