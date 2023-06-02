import numpy as np
from pymotion.ops.forward_kinematics import fk
from pymotion.ops.forward_kinematics_torch import fk as fk_torch
import pymotion.rotations.quat as quat
import pymotion.rotations.quat_torch as quat_torch
from numpy.testing import assert_allclose
import torch


class TestFK:
    atol = 1e-6
    low_atol = 1e-3  # for those operations that are not as precise

    def test_fk(self):
        # Test with a simple chain
        offsets = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
            ]
        ).astype(np.float32)
        parents = np.array([0, 0, 1])
        global_pos = np.array([[0, 0, 0], [1, 1, 1]]).astype(np.float32)
        # Test with identity rotation
        rot = np.array(
            [
                [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            ]
        ).astype(np.float32)
        ground_truth_pos = np.array(
            [
                [[0, 0, 0], [0, 0, 1], [0, 0, 3]],
                [[1, 1, 1], [1, 1, 2], [1, 1, 4]],
            ]
        ).astype(np.float32)
        ground_truth_rotmats = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ).astype(np.float32)
        ground_truth_rotmats = np.tile(
            ground_truth_rotmats, (2, 3, 1, 1)
        )  # (..., 3, 3)
        positions, rotmats = fk(rot, global_pos, offsets, parents)
        assert_allclose(positions, ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats, ground_truth_rotmats, atol=self.atol)
        positions, rotmats = fk_torch(
            torch.from_numpy(rot),
            torch.from_numpy(global_pos),
            torch.from_numpy(offsets),
            torch.from_numpy(parents),
        )
        assert_allclose(positions.numpy(), ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats.numpy(), ground_truth_rotmats, atol=self.atol)

        # Test with non-identity rotation
        def x_matrix(angle):
            return [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]

        def y_matrix(angle):
            return [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]

        def z_matrix(angle):
            return [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]

        rot = np.array(
            [
                [
                    x_matrix(np.pi / 2),
                    y_matrix(np.pi / 2),
                    z_matrix(np.pi / 2),
                ],
                [
                    y_matrix(np.pi / 4),
                    z_matrix(np.pi / 4),
                    x_matrix(np.pi / 4),
                ],
            ]
        )
        ground_truth_pos = np.array(
            [
                [[0, 0, 0], [0, -1, 0], [2, -1, 0]],
                [[1, 1, 1], [1.707107, 1, 1.707107], [3.12132, 1, 3.12132]],
            ]
        ).astype(np.float32)
        ground_truth_rotmats = np.array(
            [
                [
                    [
                        [1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0],
                    ],
                    [
                        [0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0],
                    ],
                    [
                        [0, 0, 1],
                        [0, -1, 0],
                        [1, 0, 0],
                    ],
                ],
                [
                    [
                        [0.7071068, 0, 0.7071068],
                        [0, 1, 0],
                        [-0.7071068, 0, 0.7071068],
                    ],
                    [
                        [0.5, -0.5, 0.7071068],
                        [0.7071068, 0.7071068, 0],
                        [-0.5, 0.5, 0.7071068],
                    ],
                    [
                        [0.5, 0.1464466, 0.8535535],
                        [0.7071069, 0.5, -0.5],
                        [-0.5, 0.8535535, 0.1464465],
                    ],
                ],
            ]
        ).astype(np.float32)
        positions, rotmats = fk(quat.from_matrix(rot), global_pos, offsets, parents)
        assert_allclose(positions, ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats, ground_truth_rotmats, atol=self.atol)
        positions, rotmats = fk_torch(
            quat_torch.from_matrix(torch.from_numpy(rot)),
            torch.from_numpy(global_pos),
            torch.from_numpy(offsets),
            torch.from_numpy(parents),
        )
        assert_allclose(positions.numpy(), ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats.numpy(), ground_truth_rotmats, atol=self.atol)
        # Test multidimension
        rot = np.tile(rot, (4, 3, 2, 1, 1, 1))
        global_pos = np.tile(global_pos, (4, 3, 2, 1))
        ground_truth_pos = np.tile(ground_truth_pos, (4, 3, 2, 1, 1))
        ground_truth_rotmats = np.tile(ground_truth_rotmats, (4, 3, 2, 1, 1, 1))
        positions, rotmats = fk(quat.from_matrix(rot), global_pos, offsets, parents)
        assert_allclose(positions, ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats, ground_truth_rotmats, atol=self.atol)
        positions, rotmats = fk_torch(
            quat_torch.from_matrix(torch.from_numpy(rot)),
            torch.from_numpy(global_pos),
            torch.from_numpy(offsets),
            torch.from_numpy(parents),
        )
        assert_allclose(positions.numpy(), ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats.numpy(), ground_truth_rotmats, atol=self.atol)
