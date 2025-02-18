import pymotion.rotations.quat as quat
import pymotion.rotations.quat_torch as quat_torch
import numpy as np
import torch
from numpy.testing import assert_allclose


# This code lets you see the effect of unrolling a quaternion
# import matplotlib.pyplot as plt  # uncomment this line to see the effect of unrolling a quaternion
# def test_unroll(qs, joint=0):
#     fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
#     ax0.plot(range(qs.shape[-3]), qs[..., joint, 0], label="w")
#     ax0.plot(range(qs.shape[-3]), qs[..., joint, 1], label="x")
#     ax0.plot(range(qs.shape[-3]), qs[..., joint, 2], label="y")
#     ax0.plot(range(qs.shape[-3]), qs[..., joint, 3], label="z")
#     ax0.set_title("Input")
#     qs_unrolled = quat.unroll(qs, axis=-3)
#     ax1.plot(range(qs.shape[-3]), qs_unrolled[..., joint, 0], label="w")
#     ax1.plot(range(qs.shape[-3]), qs_unrolled[..., joint, 1], label="x")
#     ax1.plot(range(qs.shape[-3]), qs_unrolled[..., joint, 2], label="y")
#     ax1.plot(range(qs.shape[-3]), qs_unrolled[..., joint, 3], label="z")
#     ax1.set_title("Unrolled")
#     fig.legend()
#     plt.show()


class TestQuat:
    atol = 1e-6
    low_atol = 1e-3  # for those operations that are not as precise

    def test_angle_axis(self):
        n = 100
        # create n random axis of rotation with length=1
        axis = np.random.rand(1, 2, 3, n, 3)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        axis_t = torch.from_numpy(axis)
        # create n random angles (scales)
        angle = (np.random.rand(1, 2, 3, n) * 2 * np.pi)[..., np.newaxis]
        angle_t = torch.from_numpy(angle)
        # create quaternions from axis and angle
        q = quat.from_angle_axis(angle, axis)
        q_t = quat_torch.from_angle_axis(angle_t, axis_t)
        # check if the quaternions are unit length
        assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=self.atol)
        assert_allclose(torch.norm(q_t, dim=-1).numpy(), 1, atol=self.atol)
        # check if inverse operations produce the same result
        t_angle, t_axis = quat.to_angle_axis(q)
        t_angle_t, t_axis_t = quat_torch.to_angle_axis(q_t)
        assert_allclose(t_angle, angle, atol=self.atol)
        assert_allclose(t_axis, axis, atol=self.atol)
        assert_allclose(t_angle_t.numpy(), angle, atol=self.atol)
        assert_allclose(t_axis_t.numpy(), axis, atol=self.atol)
        # simple samples
        axis = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        axis_t = torch.from_numpy(axis)
        angle = np.array([0, np.pi / 2, np.pi / 4, np.pi])[..., np.newaxis]
        angle_t = torch.from_numpy(angle)
        q = quat.from_angle_axis(angle, axis)
        q_t = quat_torch.from_angle_axis(angle_t, axis_t)
        ground_truth = np.array(
            [
                [1, 0, 0, 0],
                [0.70710678, 0.70710678, 0, 0],
                [0.92387953, 0, 0.38268343, 0],
                [0, 0, 0, 1],
            ]
        )
        assert_allclose(q, ground_truth, atol=self.atol)
        assert_allclose(q_t.numpy(), ground_truth, atol=self.atol)
        q = ground_truth
        q_t = torch.from_numpy(ground_truth)
        t_angle, t_axis = quat.to_angle_axis(q)
        t_angle_t, t_axis_t = quat_torch.to_angle_axis(q_t)
        assert_allclose(t_angle, angle, atol=self.atol)
        assert_allclose(t_axis, axis, atol=self.atol)
        assert_allclose(t_angle_t.numpy(), angle, atol=self.atol)
        assert_allclose(t_axis_t.numpy(), axis, atol=self.atol)

    def test_scaled_angle_axis(self):
        n = 100
        # create n random axis of rotation with length=1
        axis = np.random.rand(1, 2, 3, n, 3)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        # create n random angles (scales)
        angle = (np.random.rand(1, 2, 3, n) * 2 * np.pi)[..., np.newaxis]
        # create scaled axis angle
        axis *= angle
        axis_t = torch.from_numpy(axis)
        # create quaternions from axis and angle
        q = quat.from_scaled_angle_axis(axis)
        q_t = quat_torch.from_scaled_angle_axis(axis_t)
        # check if the quaternions are unit length
        assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=self.atol)
        assert_allclose(torch.norm(q_t, dim=-1), 1, atol=self.atol)
        # check if inverse operations produce the same result
        t_axis = quat.to_scaled_angle_axis(q)
        t_axis_t = quat_torch.to_scaled_angle_axis(q_t)
        assert_allclose(t_axis, axis, atol=self.atol)
        assert_allclose(t_axis_t, axis, atol=self.atol)
        # simple samples
        axis = np.array([[np.pi / 2, 0, 0], [0, np.pi / 4, 0], [0, 0, np.pi]])
        axis_t = torch.from_numpy(axis)
        q = quat.from_scaled_angle_axis(axis)
        q_t = quat_torch.from_scaled_angle_axis(axis_t)
        ground_truth = np.array(
            [
                [0.70710678, 0.70710678, 0, 0],
                [0.92387953, 0, 0.38268343, 0],
                [0, 0, 0, 1],
            ]
        )
        assert_allclose(q, ground_truth, atol=self.atol)
        assert_allclose(q_t, torch.from_numpy(ground_truth), atol=self.atol)
        q = ground_truth
        q_t = torch.from_numpy(ground_truth)
        t_axis = quat.to_scaled_angle_axis(q)
        t_axis_t = quat_torch.to_scaled_angle_axis(q_t)
        assert_allclose(t_axis, axis, atol=self.atol)
        assert_allclose(t_axis_t.numpy(), axis, atol=self.atol)

    def test_euler(self):
        n = 100
        # create n random euler angles
        euler = np.random.rand(1, 2, 3, n, 3) * 2 * np.pi
        euler_t = torch.from_numpy(euler)
        # create n random rotation orders
        orders = [
            ["x", "y", "z"],
            ["x", "z", "y"],
            ["y", "x", "z"],
            ["y", "z", "x"],
            ["z", "x", "y"],
            ["z", "y", "x"],
        ]
        indices = np.random.randint(0, len(orders), (1, 2, 3, n, 1))
        order = np.apply_along_axis(lambda x: orders[x.item()], -1, indices)
        # create quaternions from euler angles
        q = quat.from_euler(euler, order)
        q_t = quat_torch.from_euler(euler_t, order)
        # check if the quaternions are unit length
        assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=self.atol)
        assert_allclose(torch.norm(q_t, dim=-1), 1, atol=self.atol)
        # check if inverse operations produce the same result
        t_euler = quat.to_euler(q, order)
        t_euler_t = quat_torch.to_euler(q_t, order)
        t2_euler = quat.to_euler(quat.from_euler(t_euler, order), order)  # have euler in the same "form"
        t2_euler_t = quat_torch.to_euler(quat_torch.from_euler(t_euler_t, order), order)
        assert_allclose(t_euler, t2_euler, atol=self.low_atol)
        assert_allclose(t_euler_t.numpy(), t2_euler_t.numpy(), atol=self.low_atol)
        # simple samples
        euler = np.array([[np.pi / 2, 0, 0], [0, np.pi / 4, 0], [0, 0, np.pi]])
        euler_t = torch.from_numpy(euler)
        order = np.array([["x", "y", "z"], ["x", "z", "y"], ["y", "x", "z"]])
        q = quat.from_euler(euler, order)
        q_t = quat_torch.from_euler(euler_t, order)
        ground_truth = np.array(
            [
                [0.70710678, 0.70710678, 0, 0],
                [0.92387953, 0, 0, 0.38268343],
                [0, 0, 0, 1],
            ]
        )
        assert_allclose(q, ground_truth, atol=self.atol)
        assert_allclose(q_t, torch.from_numpy(ground_truth), atol=self.atol)
        q = ground_truth
        q_t = torch.from_numpy(ground_truth)
        t_euler = quat.to_euler(q, order)
        t_euler_t = quat_torch.to_euler(q_t, order)
        assert_allclose(t_euler, euler, atol=self.atol)
        assert_allclose(t_euler_t.numpy(), euler, atol=self.atol)

    def test_matrix(self):
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
        # check if inverse operations produce the same result
        t_q = quat.from_matrix(r)
        t_q_t = quat_torch.from_matrix(r_t)
        # check if the quaternions are unit length
        assert_allclose(np.linalg.norm(t_q, axis=-1), 1, atol=self.atol)
        assert_allclose(torch.norm(t_q_t, dim=-1), 1, atol=self.atol)
        # check if inverse operations produce the same result
        q_id = np.tile(np.array([1, 0, 0, 0]), (1, 1, 1, 1, 1))
        t_q = np.concatenate([q_id, t_q], axis=-2)
        t_q_t = np.concatenate([q_id, t_q_t.numpy()], axis=-2)
        q = np.concatenate([q_id, q], axis=-2)
        t_q = quat.unroll(t_q, axis=-2)
        t_q_t = quat.unroll(t_q_t, axis=-2)
        q = quat.unroll(q, axis=-2)
        assert_allclose(t_q, q, atol=self.atol)
        assert_allclose(t_q_t, q, atol=self.atol)
        # simple samples
        q = np.array(
            [
                [0.70710678, 0.70710678, 0, 0],
                [0.92387953, 0, 0.38268343, 0],
                [0, 0, 0, 1],
            ]
        )
        q_t = torch.from_numpy(q)
        r = quat.to_matrix(q)
        r_t = quat_torch.to_matrix(q_t)
        ground_truth = np.array(
            [
                [
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0],
                ],
                [
                    [0.70710678, 0, 0.70710678],
                    [0, 1, 0],
                    [-0.70710678, 0, 0.70710678],
                ],
                [
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                ],
            ]
        )
        assert_allclose(r, ground_truth, atol=self.atol)
        assert_allclose(r_t.numpy(), ground_truth, atol=self.atol)
        r = ground_truth
        r_t = torch.from_numpy(ground_truth)
        t_q = quat.from_matrix(r)
        t_q_t = quat_torch.from_matrix(r_t)
        assert_allclose(t_q, q, atol=self.atol)
        assert_allclose(t_q_t.numpy(), q, atol=self.atol)

    def test_mul_vec(self):
        # quaternions
        q = np.array(
            [
                [0.70710678, 0.70710678, 0, 0],  # axis: (1,0,0), angle: pi/2
                [0.92387953, 0, 0.38268343, 0],  # axis: (0,1,0), angle: pi/4
                [0, 0, 0, 1],  # axis: (0,0,1), angle: pi
            ]
        )
        q_t = torch.from_numpy(q)
        # vectors
        v = np.array([[0, 2, 0], [0, 0, -4], [1, 2, 0]])
        v_t = torch.from_numpy(v)
        # ground truth rotated vectors
        ground_truth = np.array(
            [
                [0, 0, 2],
                [-2.828427, 0, -2.828427],
                [-1, -2, 0],
            ]
        )
        # rotate vectors
        v_rot = quat.mul_vec(q, v)
        v_rot_t = quat_torch.mul_vec(q_t, v_t)
        assert_allclose(v_rot, ground_truth, atol=self.atol)
        assert_allclose(v_rot_t.numpy(), ground_truth, atol=self.atol)
        # rotate vectors with inverse quaternion
        q_inv = quat.inverse(q)
        q_inv_t = quat_torch.inverse(q_t)
        v_rot = quat.mul_vec(q_inv, v_rot)
        v_rot_t = quat_torch.mul_vec(q_inv_t, v_rot_t)
        assert_allclose(v_rot, v, atol=self.atol)
        assert_allclose(v_rot_t.numpy(), v, atol=self.atol)

    def test_mul(self):
        # quaternions
        q1 = np.array(
            [
                [0.70710678, 0.70710678, 0, 0],  # axis: (1,0,0), angle: pi/2
                [0.92387953, 0, 0.38268343, 0],  # axis: (0,1,0), angle: pi/4
                [0, 0, 0, 1],  # axis: (0,0,1), angle: pi
            ]
        )
        q1_t = torch.from_numpy(q1)
        q2 = np.array(
            [
                [0, 0, 0, 1],  # axis: (0,0,1), angle: pi
                [0.70710678, 0.70710678, 0, 0],  # axis: (1,0,0), angle: pi/2
                [0.92387953, 0, 0.38268343, 0],  # axis: (0,1,0), angle: pi/4
            ]
        )
        q2_t = torch.from_numpy(q2)
        # ground truth
        ground_truth_1 = np.array(
            [
                [0, 0, -0.70710678, 0.70710678],
                [0.65328148, 0.65328148, 0.27059805, -0.27059805],
                [0, -0.3826834, 0, 0.92387953],
            ]
        )
        ground_truth_2 = np.array(
            [
                [0, 0, 0.70710678, 0.70710678],
                [0.65328148, 0.65328148, 0.27059805, 0.27059805],
                [0, 0.3826834, 0, 0.92387953],
            ]
        )
        # multiply quaternions
        q1_q2 = quat.mul(q1, q2)
        q1_q2_t = quat_torch.mul(q1_t, q2_t)
        assert_allclose(q1_q2, ground_truth_1, atol=self.atol)
        assert_allclose(q1_q2_t.numpy(), ground_truth_1, atol=self.atol)
        q2_q1 = quat.mul(q2, q1)
        q2_q1_t = quat_torch.mul(q2_t, q1_t)
        assert_allclose(q2_q1, ground_truth_2, atol=self.atol)
        assert_allclose(q2_q1_t.numpy(), ground_truth_2, atol=self.atol)

    def test_length(self):
        n = 100
        q = np.array(
            [
                [0.70710678, 0.70710678, 0, 0],  # axis: (1,0,0), angle: pi/2
                [0.92387953, 0, 0.38268343, 0],  # axis: (0,1,0), angle: pi/4
                [0, 0, 0, 1],  # axis: (0,0,1), angle: pi
            ]
        )
        qs = np.tile(q, (n, 1)).reshape(-1, 4)
        # lengths
        lengths = np.random.rand(n * 3, 1)
        qs = qs * lengths
        # pytorch
        q_t = torch.from_numpy(qs)
        # compute length
        t_lengths = quat.length(qs)
        t_lengths_t = quat_torch.length(q_t)
        assert_allclose(t_lengths, lengths[:, 0], atol=self.atol)
        assert_allclose(t_lengths_t.numpy(), lengths[:, 0], atol=self.atol)

    def test_inverse(self):
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
        q_id = np.tile(np.array([1, 0, 0, 0]), (1, 1, 1, n, 1))
        # compute inverse
        q_inv = quat.inverse(q)
        q_inv_t = quat_torch.inverse(q_t)
        # compute inverse with inverse quaternion
        q_inv_inv = quat.inverse(q_inv)
        q_inv_inv_t = quat_torch.inverse(q_inv_t)
        # check if inverse is correct
        assert_allclose(q, q_inv_inv, atol=self.atol)
        assert_allclose(q_t, q_inv_inv_t.numpy(), atol=self.atol)
        # check if multiply inverse is identity
        assert_allclose(quat.mul(q, q_inv), q_id, atol=self.atol)
        assert_allclose(quat_torch.mul(q_t, q_inv_t).numpy(), q_id, atol=self.atol)

    def test_normalize(self):
        n = 100
        q = np.array(
            [
                [0.70710678, 0.70710678, 0, 0],  # axis: (1,0,0), angle: pi/2
                [0.92387953, 0, 0.38268343, 0],  # axis: (0,1,0), angle: pi/4
                [0, 0, 0, 1],  # axis: (0,0,1), angle: pi
            ]
        )
        qs = np.tile(q, (n, 1)).reshape(-1, 4)
        # lengths
        lengths = np.random.rand(n * 3, 1)
        qs = qs * lengths
        # pytorch
        q_t = torch.from_numpy(qs)
        # normalize
        qs_norm = quat.normalize(qs)
        qs_norm_t = quat_torch.normalize(q_t)
        # check if lengths are 1
        t1 = np.linalg.norm(qs_norm, axis=-1)
        t2 = np.linalg.norm(qs_norm_t.numpy(), axis=-1)
        assert_allclose(t1, np.ones_like(t1), atol=self.low_atol)
        assert_allclose(t2, np.ones_like(t2), atol=self.low_atol)

    def test_slerp(self):
        axis_1 = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
            ]
        )
        axis_2 = np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        angle_1 = np.array(
            [
                [0],
                [np.pi / 2],
                [np.pi],
            ]
        )
        angle_2 = np.array(
            [
                [np.pi / 2],
                [0],
                [np.pi],
            ]
        )
        t = np.array(
            [
                [0],
                [0.5],
                [1],
            ]
        )
        t_2 = np.array(
            [
                [-1],
                [2],
                [0.25],
            ]
        )
        t_3 = 0.75
        q1 = quat.from_angle_axis(angle_1, axis_1)
        q2 = quat.from_angle_axis(angle_2, axis_2)
        gt = np.array([[1, 0, 0, 0], [0.92387953, 0, 0, 0.38268343], [0, 0, 1, 0]])
        gt_2 = np.array(
            [
                [0.70710678, 0, 0, -0.70710678],
                [0.70710678, 0, 0, -0.70710678],
                [0, 0, 0.38268343, 0.92387953],
            ]
        )
        gt_3 = np.array(
            [
                [0.83146961, 0, 0, 0.5555702],
                [0.98078528, 0, 0, 0.1950903],
                [0, 0, 0.9238795, 0.3826834],
            ]
        )
        assert_allclose(quat.slerp(q1, q2, t), gt, atol=self.atol)
        assert_allclose(
            quat_torch.slerp(torch.from_numpy(q1), torch.from_numpy(q2), torch.from_numpy(t)).numpy(),
            gt,
            atol=self.atol,
        )
        assert_allclose(quat.slerp(q1, q2, t_2), gt_2, atol=self.atol)
        assert_allclose(
            quat_torch.slerp(torch.from_numpy(q1), torch.from_numpy(q2), torch.from_numpy(t_2)).numpy(),
            gt_2,
            atol=self.atol,
        )
        assert_allclose(quat.slerp(q1, q2, t_3), gt_3, atol=self.atol)
        assert_allclose(
            quat_torch.slerp(torch.from_numpy(q1), torch.from_numpy(q2), t_3).numpy(),
            gt_3,
            atol=self.atol,
        )
        assert_allclose(
            quat.slerp(
                q1[np.newaxis, np.newaxis, ...],
                q2[np.newaxis, np.newaxis, ...],
                t_2[np.newaxis, np.newaxis, ...],
            ),
            gt_2[np.newaxis, np.newaxis, ...],
            atol=self.atol,
        )
        assert_allclose(
            quat_torch.slerp(
                torch.from_numpy(q1[np.newaxis, np.newaxis, ...]),
                torch.from_numpy(q2[np.newaxis, np.newaxis, ...]),
                torch.from_numpy(t_2[np.newaxis, np.newaxis, ...]),
            ).numpy(),
            gt_2[np.newaxis, np.newaxis, ...],
            atol=self.atol,
        )

        q1 = quat.from_angle_axis(np.array([np.pi / 2]), np.array([0, 1, 1]) / np.sqrt(2))
        q2 = -q1.copy()
        gt = q1
        assert_allclose(quat.slerp(q1, q2, 0.5), gt, atol=self.low_atol)
        assert_allclose(
            quat_torch.slerp(torch.from_numpy(q1), torch.from_numpy(q2), 0.5).numpy(),
            gt,
            atol=self.low_atol,
        )
        gt = np.array([0.5, 0.0, 0.353553, 0.353553])
        assert_allclose(quat.slerp(q1, q2, 0.25, shortest=False), gt, atol=self.low_atol)
        assert_allclose(
            quat_torch.slerp(torch.from_numpy(q1), torch.from_numpy(q2), 0.25, shortest=False).numpy(),
            gt,
            atol=self.low_atol,
        )
