import pymotion.rotations.quat as quat
import pymotion.rotations.quat_torch as quat_torch
import pymotion.rotations.ortho6d as sixd
import pymotion.rotations.ortho6d_torch as sixd_torch
import numpy as np
import torch
from numpy.testing import assert_allclose


class TestOrtho6D:
    atol = 1e-6
    low_atol = 1e-3  # for those operations that are not as precise

    def test_quat_and_mat(self):
        n = 100
        # create n random axis of rotation with length=1
        axis = np.random.rand(1, 1, 1, n, 3)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        # create n random angles (scales)
        angle = (np.random.rand(1, 1, 1, n) * 2 * np.pi)[..., np.newaxis]
        # create scaled axis angle
        axis *= angle
        axis_t = torch.from_numpy(axis)
        # create quaternions from axis and angle
        q = quat.from_scaled_angle_axis(axis)
        q_t = quat_torch.from_scaled_angle_axis(axis_t)
        # create rotation matrices from quaternions
        r = quat.to_matrix(q)
        r_t = quat_torch.to_matrix(q_t)
        # create ortho6D from quaternions
        o = sixd.from_quat(q)
        o_t = sixd_torch.from_quat(q_t)
        # create ortho6D from rotation matrices
        o2 = sixd.from_matrix(r)
        o2_t = sixd_torch.from_matrix(r_t)
        # create quaternions from ortho6D
        q2 = sixd.to_quat(o)
        q2_t = sixd_torch.to_quat(o_t)
        q_id = np.tile(np.array([1, 0, 0, 0]), (1, 1, 1, 1, 1))
        t_q = np.concatenate([q_id, q2], axis=-2)
        t_q_t = torch.cat([torch.from_numpy(q_id), q2_t], dim=-2)
        q = np.concatenate([q_id, q], axis=-2)
        q_t = torch.cat([torch.from_numpy(q_id), q_t], dim=-2)
        t_q = quat.unroll(t_q, axis=-2)
        t_q_t = quat_torch.unroll(t_q_t, dim=-2)
        q = quat.unroll(q, axis=-2)
        q_t = quat_torch.unroll(q_t, dim=-2)
        # create rotation matrices from ortho6D
        r2 = sixd.to_matrix(o)
        r2_t = sixd_torch.to_matrix(o_t)
        # compare
        assert_allclose(q, t_q, atol=self.atol)
        assert_allclose(q_t.numpy(), t_q_t.numpy(), atol=self.atol)
        assert_allclose(r, r2, atol=self.atol)
        assert_allclose(r_t.numpy(), r2_t.numpy(), atol=self.atol)
        assert_allclose(o, o2, atol=self.atol)
        assert_allclose(o_t.numpy(), o2_t.numpy(), atol=self.atol)
