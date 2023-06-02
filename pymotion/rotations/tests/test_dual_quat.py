import pymotion.rotations.quat as quat
import pymotion.rotations.quat_torch as quat_torch
import pymotion.rotations.dual_quat as dual_quat
import pymotion.rotations.dual_quat_torch as dual_quat_torch
import numpy as np
import torch
from numpy.testing import assert_allclose


class TestDualQuat:
    atol = 1e-6
    low_atol = 1e-3  # for those operations that are not as precise

    def test_from_to(self):
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
        # create n random translations
        t = np.random.rand(1, 1, 1, n, 3)
        t_t = torch.from_numpy(t)
        # create dual quaternions from quaternions and translations
        dq = dual_quat.from_rotation_translation(q, t)
        dq_t = dual_quat_torch.from_rotation_translation(q_t, t_t)
        # convert dual quaternions back to quaternions and translations
        q2, t2 = dual_quat.to_rotation_translation(dq)
        q2_t, t2_t = dual_quat_torch.to_rotation_translation(dq_t)
        # compare quaternions and translations
        assert_allclose(q, q2, atol=self.atol)
        assert_allclose(q_t, q2_t, atol=self.atol)
        assert_allclose(t, t2, atol=self.atol)
        assert_allclose(t_t, t2_t, atol=self.atol)
        # create dual quaternions from translation only
        dq = dual_quat.from_translation(t)
        dq_t = dual_quat_torch.from_translation(t_t)
        # convert dual quaternions back to quaternions and translations
        _, t2 = dual_quat.to_rotation_translation(dq)
        _, t2_t = dual_quat_torch.to_rotation_translation(dq_t)
        # compare translations
        assert_allclose(t, t2, atol=self.atol)
        assert_allclose(t_t, t2_t, atol=self.atol)

    def test_normalize(self):
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
        # create n random translations
        t = np.random.rand(1, 1, 1, n, 3)
        t_t = torch.from_numpy(t)
        # create dual quaternions from quaternions and translations
        dq = dual_quat.from_rotation_translation(q, t)
        dq_t = dual_quat_torch.from_rotation_translation(q_t, t_t).float()
        # normalize dual quaternions
        dq = dual_quat.normalize(dq)
        dq_t = dual_quat_torch.normalize(dq_t)
        # check if unit
        assert np.all(dual_quat.is_unit(dq))
        assert torch.all(dual_quat_torch.is_unit(dq_t))
