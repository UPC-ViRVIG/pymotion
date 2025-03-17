import torch
import numpy as np
import pymotion.ops.skeleton as skeleton
import pymotion.ops.skeleton_torch as skeleton_torch
import pymotion.rotations.quat as quat
import pymotion.rotations.quat_torch as quat_torch
import pymotion.rotations.dual_quat as dual_quat
import pymotion.rotations.dual_quat_torch as dual_quat_torch
from pymotion.ops.skeleton import fk
from pymotion.ops.skeleton_torch import fk as fk_torch
from pymotion.ops.skeleton import from_global_rotations
from pymotion.ops.skeleton_torch import from_global_rotations as from_global_rotations_torch
from pymotion.ops.skeleton import from_root_positions
from pymotion.ops.skeleton_torch import from_root_positions as from_root_positions_torch
from pymotion.io.bvh import BVH
from numpy.testing import assert_allclose


class TestSkeleton:
    atol = 1e-6
    low_atol = 1e-3  # for those operations that are not as precise
    low_atol_2 = 1e-2

    def test_dq(self):
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
        ground_truth_offsets = np.array(
            [
                [[0, 0, 0], [0, 0, 1], [0, 0, 3]],
                [[1, 1, 1], [0, 0, 1], [0, 0, 3]],
            ]
        ).astype(np.float32)
        ground_truth_rot = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ).astype(np.float32)
        ground_truth_rot = np.tile(ground_truth_rot, (2, 3, 1, 1))  # (..., 3, 3)
        dqs = skeleton.to_root_dual_quat(rot, global_pos, parents, offsets)
        rots_dq, trans_dq = dual_quat.to_rotation_translation(dqs)
        assert_allclose(rots_dq, quat.from_matrix(ground_truth_rot), atol=self.atol)
        assert_allclose(trans_dq, ground_truth_offsets, atol=self.atol)
        dqs_t = skeleton_torch.to_root_dual_quat(
            torch.from_numpy(rot),
            torch.from_numpy(global_pos),
            torch.from_numpy(parents),
            torch.from_numpy(offsets),
        )
        rots_dq_t, trans_dq_t = dual_quat_torch.to_rotation_translation(dqs_t)
        assert_allclose(rots_dq_t.numpy(), quat.from_matrix(ground_truth_rot), atol=self.atol)
        assert_allclose(trans_dq_t.numpy(), ground_truth_offsets, atol=self.atol)
        # inverse
        trans_sk, rots_sk = skeleton.from_root_dual_quat(dqs, parents)
        assert_allclose(rots_sk, rot, atol=self.atol)
        assert_allclose(
            trans_sk[:, 1:, :],
            np.tile(offsets[1:, :], (trans_sk.shape[0], 1, 1)),
            atol=self.atol,
        )
        assert_allclose(trans_sk[:, 0, :], global_pos, atol=self.atol)
        trans_sk_t, rots_sk_t = skeleton_torch.from_root_dual_quat(dqs_t, torch.from_numpy(parents))
        assert_allclose(rots_sk_t.numpy(), rot, atol=self.atol)
        assert_allclose(
            trans_sk_t[:, 1:, :].numpy(),
            np.tile(offsets[1:, :], (trans_sk.shape[0], 1, 1)),
            atol=self.atol,
        )
        assert_allclose(trans_sk_t[:, 0, :].numpy(), global_pos, atol=self.atol)

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
        ground_truth_offsets = np.array(
            [
                [[0, 0, 0], [0, 0, 1], [2, 0, 1]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 3]],
            ]
        ).astype(np.float32)
        ground_truth_rot = np.array(
            [
                [
                    [
                        [1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0],
                    ],
                    [
                        [0, 0, 1],
                        [0, 1, 0],
                        [-1, 0, 0],
                    ],
                    [
                        [0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0],
                    ],
                ],
                [
                    [
                        [0.7071068, 0, 0.7071068],
                        [0, 1, 0],
                        [-0.7071068, 0, 0.7071068],
                    ],
                    [
                        [0.7071068, -0.7071068, 0],
                        [0.7071068, 0.7071068, 0],
                        [0, 0, 1],
                    ],
                    [
                        [0.7071068, -0.5, 0.5],
                        [0.7071068, 0.5, -0.5],
                        [0, 0.7071068, 0.7071068],
                    ],
                ],
            ]
        ).astype(np.float32)
        dqs = skeleton.to_root_dual_quat(quat.from_matrix(rot), global_pos, parents, offsets)
        rots_dq, trans_dq = dual_quat.to_rotation_translation(dqs)
        assert_allclose(rots_dq, quat.from_matrix(ground_truth_rot), atol=self.atol)
        assert_allclose(trans_dq[:, 1:, :], ground_truth_offsets[:, 1:, :], atol=self.atol)
        assert_allclose(trans_dq[:, 0, :], global_pos, atol=self.atol)
        dqs_t = skeleton_torch.to_root_dual_quat(
            torch.from_numpy(quat.from_matrix(rot)),
            torch.from_numpy(global_pos),
            torch.from_numpy(parents),
            torch.from_numpy(offsets),
        )
        rots_dq_t, trans_dq_t = dual_quat_torch.to_rotation_translation(dqs_t)
        assert_allclose(rots_dq_t.numpy(), quat.from_matrix(ground_truth_rot), atol=self.atol)
        assert_allclose(trans_dq_t.numpy()[:, 1:, :], ground_truth_offsets[:, 1:, :], atol=self.atol)
        assert_allclose(trans_dq_t.numpy()[:, 0, :], global_pos, atol=self.atol)
        # inverse
        trans_sk, rots_sk = skeleton.from_root_dual_quat(dqs, parents)
        assert_allclose(rots_sk, quat.from_matrix(rot), atol=self.atol)
        assert_allclose(
            trans_sk[:, 1:, :],
            np.tile(offsets[1:, :], (trans_sk.shape[0], 1, 1)),
            atol=self.atol,
        )
        assert_allclose(trans_sk[:, 0, :], global_pos, atol=self.atol)
        trans_sk_t, rots_sk_t = skeleton_torch.from_root_dual_quat(dqs_t, torch.from_numpy(parents))
        assert_allclose(rots_sk_t.numpy(), quat.from_matrix(rot), atol=self.atol)
        assert_allclose(
            trans_sk_t[:, 1:, :].numpy(),
            np.tile(offsets[1:, :], (trans_sk.shape[0], 1, 1)),
            atol=self.atol,
        )
        assert_allclose(trans_sk_t[:, 0, :].numpy(), global_pos, atol=self.atol)
        # Test multidimension
        rot = np.tile(rot, (4, 3, 2, 1, 1, 1))
        global_pos = np.tile(global_pos, (4, 3, 2, 1))
        ground_truth_rot = rot.copy()
        dqs = skeleton.to_root_dual_quat(quat.from_matrix(rot), global_pos, parents, offsets)
        _, rots_dq = skeleton.from_root_dual_quat(dqs, parents)
        assert_allclose(rots_dq, quat.from_matrix(ground_truth_rot), atol=self.atol)
        dqs_t = skeleton_torch.to_root_dual_quat(
            torch.from_numpy(quat.from_matrix(rot)),
            torch.from_numpy(global_pos),
            torch.from_numpy(parents),
            torch.from_numpy(offsets),
        )
        _, rots_dq = skeleton_torch.from_root_dual_quat(dqs_t, parents)
        assert_allclose(rots_dq.numpy(), quat.from_matrix(ground_truth_rot), atol=self.atol)

    def test_from_positions(self):
        bvh = BVH()
        bvh.load("test.bvh")
        local_rotations, local_positions, parents, offsets, _, _ = bvh.get_data()

        pos, _ = fk(local_rotations, np.zeros((local_positions.shape[0], 3)), offsets, parents)

        pred_rots = from_root_positions(pos, parents, offsets)
        pred_rots_t = from_root_positions_torch(
            torch.from_numpy(pos).float(), torch.from_numpy(parents), torch.from_numpy(offsets).float()
        )

        assert_allclose(pred_rots, pred_rots_t.numpy(), atol=self.low_atol_2)

        bvh.set_data(pred_rots, local_positions)

        pred_pos, _ = fk(pred_rots, np.zeros((local_positions.shape[0], 3)), offsets, parents)

        assert_allclose(pos, pred_pos, atol=self.low_atol_2)

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
        ground_truth_rotmats = np.tile(ground_truth_rotmats, (2, 3, 1, 1))  # (..., 3, 3)
        positions, rotmats = fk(rot, global_pos, np.tile(offsets, (2, 1, 1)), parents)
        positions, rotmats = fk(rot, global_pos, offsets, parents)
        local_rots = from_global_rotations(quat.from_matrix(rotmats), parents)
        assert_allclose(positions, ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats, ground_truth_rotmats, atol=self.atol)
        assert_allclose(local_rots, rot, atol=self.atol)
        positions, rotmats = fk_torch(
            torch.from_numpy(rot),
            torch.from_numpy(global_pos),
            torch.from_numpy(np.tile(offsets, (2, 1, 1))),
            torch.from_numpy(parents),
        )
        positions, rotmats = fk_torch(
            torch.from_numpy(rot),
            torch.from_numpy(global_pos),
            torch.from_numpy(offsets),
            torch.from_numpy(parents),
        )
        local_rots = from_global_rotations_torch(quat_torch.from_matrix(rotmats), torch.from_numpy(parents))
        assert_allclose(positions.numpy(), ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats.numpy(), ground_truth_rotmats, atol=self.atol)
        assert_allclose(local_rots.numpy(), rot, atol=self.atol)

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
        local_rots = from_global_rotations(quat.from_matrix(rotmats), parents)
        assert_allclose(positions, ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats, ground_truth_rotmats, atol=self.atol)
        assert_allclose(local_rots, quat.from_matrix(rot), atol=self.atol)
        positions, rotmats = fk_torch(
            quat_torch.from_matrix(torch.from_numpy(rot)),
            torch.from_numpy(global_pos),
            torch.from_numpy(offsets),
            torch.from_numpy(parents),
        )
        local_rots = from_global_rotations_torch(quat_torch.from_matrix(rotmats), torch.from_numpy(parents))
        assert_allclose(positions.numpy(), ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats.numpy(), ground_truth_rotmats, atol=self.atol)
        assert_allclose(local_rots.numpy(), quat.from_matrix(rot), atol=self.atol)
        # Test multidimension
        rot = np.tile(rot, (4, 3, 2, 1, 1, 1))
        global_pos = np.tile(global_pos, (4, 3, 2, 1))
        ground_truth_pos = np.tile(ground_truth_pos, (4, 3, 2, 1, 1))
        ground_truth_rotmats = np.tile(ground_truth_rotmats, (4, 3, 2, 1, 1, 1))
        positions, rotmats = fk(quat.from_matrix(rot), global_pos, offsets, parents)
        local_rots = from_global_rotations(quat.from_matrix(rotmats), parents)
        assert_allclose(positions, ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats, ground_truth_rotmats, atol=self.atol)
        assert_allclose(local_rots, quat.from_matrix(rot), atol=self.atol)
        positions, rotmats = fk_torch(
            quat_torch.from_matrix(torch.from_numpy(rot)),
            torch.from_numpy(global_pos),
            torch.from_numpy(offsets),
            torch.from_numpy(parents),
        )
        local_rots = from_global_rotations_torch(quat_torch.from_matrix(rotmats), torch.from_numpy(parents))
        assert_allclose(positions.numpy(), ground_truth_pos, atol=self.atol)
        assert_allclose(rotmats.numpy(), ground_truth_rotmats, atol=self.atol)
        assert_allclose(local_rots.numpy(), quat.from_matrix(rot), atol=self.atol)
