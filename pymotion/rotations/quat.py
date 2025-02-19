from __future__ import annotations
import numpy as np
import pymotion.ops.vector as vec


def from_scaled_angle_axis(scaledaxis: np.array) -> np.array:
    """
    Create a quaternion from an scaled angle-axis representation.

    Parameters
    ----------
    scaledaxis : np.array[..., [x,y,z]]
        axis [x,y,z] of rotation where magnitude is the angle of rotation

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """
    angle = np.linalg.norm(scaledaxis, axis=-1)[..., np.newaxis]
    axis = scaledaxis / angle
    return from_angle_axis(angle, axis)


def from_angle_axis(angle: np.array, axis: np.array) -> np.array:
    """
    Create a quaternion from an angle-axis representation.

    Parameters
    ----------
    angle : np.array[..., angle] in radians.
    axis : np.array[..., [x,y,z]]
        normalized axis [x,y,z] of rotation

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """
    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    return np.concatenate((c, s * axis), axis=-1)


def from_euler(euler: np.array, order: np.array) -> np.array:
    """
    Create a quaternion from an euler representation with a specified order.

    Parameters
    ----------
    euler : np.array[..., [e0, e1, e2]]
        euler angles in radians
    order : np.array[..., ['x'|'y'|'z', 'x'|'y'|'z', 'x'|'y'|'z']]
        order of the euler angles
        symmetric orders not supported (e.g., XYX).

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """

    assert (
        euler.shape[:-1] == order.shape[:-1]
    ), "euler and order must have the same shape except for the last dimension"

    axis = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }

    q0 = from_angle_axis(
        euler[..., 0:1],
        np.apply_along_axis(lambda x: axis[x.item()], -1, order[..., 0:1]),
    )
    q1 = from_angle_axis(
        euler[..., 1:2],
        np.apply_along_axis(lambda x: axis[x.item()], -1, order[..., 1:2]),
    )
    q2 = from_angle_axis(
        euler[..., 2:3],
        np.apply_along_axis(lambda x: axis[x.item()], -1, order[..., 2:3]),
    )
    return mul(q0, mul(q1, q2))


def from_matrix(rotmats: np.array) -> np.array:
    """
    Convert rotation matrices to quaternions.

    Parameters
    ----------
        rotmats: np.array[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                     [r1.x, r1.y, r1.z],
                                                     [r2.x, r2.y, r2.z]] where ri is row i.

    Returns
    -------
        quat np.array[..., [w,x,y,z]]
    """
    # Separate components
    r0c0 = rotmats[..., 0, 0]
    r0c1 = rotmats[..., 0, 1]
    r0c2 = rotmats[..., 0, 2]
    r1c0 = rotmats[..., 1, 0]
    r1c1 = rotmats[..., 1, 1]
    r1c2 = rotmats[..., 1, 2]
    r2c0 = rotmats[..., 2, 0]
    r2c1 = rotmats[..., 2, 1]
    r2c2 = rotmats[..., 2, 2]

    return normalize(
        np.where(
            (r2c2 < 0.0)[..., np.newaxis],
            np.where(
                (r0c0 > r1c1)[..., np.newaxis],
                np.concatenate(
                    [
                        (r2c1 - r1c2)[..., np.newaxis],
                        (1.0 + r0c0 - r1c1 - r2c2)[..., np.newaxis],
                        (r1c0 + r0c1)[..., np.newaxis],
                        (r0c2 + r2c0)[..., np.newaxis],
                    ],
                    axis=-1,
                ),
                np.concatenate(
                    [
                        (r0c2 - r2c0)[..., np.newaxis],
                        (r1c0 + r0c1)[..., np.newaxis],
                        (1.0 - r0c0 + r1c1 - r2c2)[..., np.newaxis],
                        (r2c1 + r1c2)[..., np.newaxis],
                    ],
                    axis=-1,
                ),
            ),
            np.where(
                (r0c0 < -r1c1)[..., np.newaxis],
                np.concatenate(
                    [
                        (r1c0 - r0c1)[..., np.newaxis],
                        (r0c2 + r2c0)[..., np.newaxis],
                        (r2c1 + r1c2)[..., np.newaxis],
                        (1.0 - r0c0 - r1c1 + r2c2)[..., np.newaxis],
                    ],
                    axis=-1,
                ),
                np.concatenate(
                    [
                        (1.0 + r0c0 + r1c1 + r2c2)[..., np.newaxis],
                        (r2c1 - r1c2)[..., np.newaxis],
                        (r0c2 - r2c0)[..., np.newaxis],
                        (r1c0 - r0c1)[..., np.newaxis],
                    ],
                    axis=-1,
                ),
            ),
        )
    )


def to_euler(quaternions: np.array, order: np.array) -> np.array:
    """
    Convert a quaternion to an intrinsic euler representation with a specified order.
    Does not detect/solve gimbal lock.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]
    order : np.array[..., ['x'|'y'|'z', 'x'|'y'|'z', 'x'|'y'|'z']]
        order of the euler angles
        symmetric orders not supported (e.g., XYX).

    Returns
    -------
    euler : np.array[..., 3]
        euler angles in radians
    """

    assert (
        quaternions.shape[:-1] == order.shape[:-1]
    ), "quaternions and order must have the same shape except for the last dimension"

    aux = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    angle_first = 2
    angle_third = 0

    i = np.apply_along_axis(lambda x: aux[x.item()], -1, order[..., 2:3])[..., np.newaxis]
    j = np.apply_along_axis(lambda x: aux[x.item()], -1, order[..., 1:2])[..., np.newaxis]
    k = np.apply_along_axis(lambda x: aux[x.item()], -1, order[..., 0:1])[..., np.newaxis]

    # check if permutation is even or odd
    sign = (i - j) * (j - k) * (k - i) // 2

    # euler angles
    euler = np.empty(quaternions.shape[:-1] + (3,))

    # permutate quaternion elements
    a = quaternions[..., 0:1] - np.take_along_axis(quaternions, j + 1, axis=-1)
    b = (
        np.take_along_axis(quaternions, i + 1, axis=-1)
        + np.take_along_axis(quaternions, k + 1, axis=-1) * sign
    )
    c = np.take_along_axis(quaternions, j + 1, axis=-1) + quaternions[..., 0:1]
    d = np.take_along_axis(quaternions, k + 1, axis=-1) * sign - np.take_along_axis(
        quaternions, i + 1, axis=-1
    )

    # compute second angle
    euler[..., 1:2] = (2 * np.arctan2(np.hypot(c, d), np.hypot(a, b))) - (np.pi / 2)

    # compute first and third angle
    half_sum = np.arctan2(b, a)
    half_diff = np.arctan2(d, c)
    euler[..., angle_first : angle_first + 1] = half_sum - half_diff
    euler[..., angle_third : angle_third + 1] = (half_sum + half_diff) * sign

    # for i in range(3):
    # if euler[..., i] < -np.pi:
    #    euler[..., i] += 2 * np.pi
    # elif euler[..., i] > np.pi:
    #    euler[..., i] -= 2 * np.pi
    euler = np.mod(euler, 2 * np.pi)

    return euler


def to_scaled_angle_axis(quaternions: np.array) -> np.array:
    """
    Quaternion to scaled axis angle representation.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    scaledaxis : np.array[..., [x,y,z]]
        axis [x,y,z] of rotation where magnitude is the angle of rotation
    """
    angle, axis = to_angle_axis(quaternions)
    return angle * axis


def to_angle_axis(quaternions: np.array) -> np.array:
    """
    Quaternion to scaled axis angle representation.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    angle: np.array[..., angle]
    axis : np.array[..., [x,y,z]]
        normalized axis [x,y,z] of rotation
    """
    w = quaternions[..., 0]
    xyz = quaternions[..., 1:]

    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(np.clip(1.0 - w * w, 0.0, 1.0))

    # Avoid division by zero when s is close to zero (identity quaternion)
    axis = np.zeros_like(xyz)
    mask = s > 1e-8
    if mask.any():
        axis[mask] = xyz[mask] / np.expand_dims(s[mask], axis=-1)

    return angle[..., np.newaxis], axis


def to_matrix(quaternions: np.array) -> np.array:
    """
    Convert rotations given as quaternions to rotation matrices.
    Parameters
    ----------
        quaternions: np.array[..., [w,x,y,z]]
    Returns
    -------
        rotmats: np.array[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                     [r1.x, r1.y, r1.z],
                                                     [r2.x, r2.y, r2.z]] where ri is row i.
    """
    qw = quaternions[..., 0]
    qx = quaternions[..., 1]
    qy = quaternions[..., 2]
    qz = quaternions[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = np.empty(quaternions.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m


def mul_vec(q: np.array, v: np.array) -> np.array:
    """
    Multiply a vector by a quaternion

    Parameters
    ----------
    q : np.array[..., [w,x,y,z]]
    v : np.array[..., [x,y,z]]

    Returns
    -------
    v: np.array[..., [x,y,z]]
    """
    t = 2.0 * _fast_cross(q[..., 1:], v)
    return v + q[..., 0][..., np.newaxis] * t + _fast_cross(q[..., 1:], t)


def mul(q0: np.array, q1: np.array) -> np.array:
    """
    Multiply two quaternions.

    Parameters
    ----------
    q0 : np.array[..., [w,x,y,z]]
    q1 : np.array[..., [w,x,y,z]]

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """
    w0, x0, y0, z0 = q0[..., 0:1], q0[..., 1:2], q0[..., 2:3], q0[..., 3:4]
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    # (w0,v0)(w1,v1) = (w0w1 - v0·v1, w0v1 + w1v0 + v0 x v1)
    return np.concatenate(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,  # w
            w0 * x1 + w1 * x0 + y0 * z1 - z0 * y1,  # x
            w0 * y1 + w1 * y0 + z0 * x1 - x0 * z1,  # y
            w0 * z1 + w1 * z0 + x0 * y1 - y0 * x1,  # z
        ),
        axis=-1,
    )


def length(quaternions: np.array) -> np.array:
    """
    Get the length or magnitude of the quaternions.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    length : np.array[...]
    """
    return np.linalg.norm(quaternions, axis=-1)


def inverse(quaternions: np.array) -> np.array:
    """
    Inverse of a quaternion.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    # for a unit quaternion the conjugate is the inverse
    # q^-1 = [q0, -q1, -q2, -q3]
    return conjugate(quaternions)


def conjugate(quaternions: np.array) -> np.array:
    """
    Compute the conjugate of a quaternion.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    return np.concatenate((quaternions[..., 0:1], -quaternions[..., 1:]), axis=-1)


def normalize(quaternions: np.array, eps: float = 1e-8) -> np.array:
    """
    Convert all quaternions to unit quatenrions.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    return quaternions / (length(quaternions)[..., np.newaxis] + eps)


def unroll(quaternions: np.array, axis: int) -> np.array:
    """
    Avoid the quaternion 'double cover' problem by picking the cover
    of the first quaternion, and then removing sudden switches
    over the cover by ensuring that each frame uses the quaternion
    closest to the one of the previous frame.

    ('double cover': same rotation can be encoded with two
    different quaternions)

    Usage example: Ensuring an animation to have quaternions
    that represent the 'shortest' rotation path. Otherwise,
    if we SLERP between poses we would get joints rotating in
    the "longest" path.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]
    axis : int
        unroll axis (e.g., frames axis)

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    r = quaternions.swapaxes(0, axis)
    # start with the second quaternion since
    # we keep the cover of the first one
    for i in range(1, r.shape[0]):
        # distance (dot product) between the previous and current quaternion
        d0 = np.sum(r[i] * r[i - 1], axis=-1)
        # distance (dot product) between the previous and flipped current quaternion
        d1 = np.sum(-r[i] * r[i - 1], axis=-1)
        # if the distance with the flipped quaternion is smaller, use it
        r[i][d0 < d1] = -r[i][d0 < d1]
    r = r.swapaxes(0, axis)
    return r


def slerp(q0: np.array, q1: np.array, t: float | np.array, shortest: bool = True) -> np.array:
    """
    Perform spherical linear interpolation (SLERP) between two unit quaternions.

    Parameters
    ----------
    q0 : np.array[..., [w,x,y,z]]
    q1 : np.array[..., [w,x,y,z]]
    t : float or np.array[..., [t]]
        Interpolation parameter between 0 and 1. At t=0, returns q0 and at t=1, returns q1.
    shorthest : bool
        Ensure the shorthest path between quaternions.

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """
    # Compute the cosine of the angle between the two vectors.
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)

    # If the dot product is negative, the quaternions
    # have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    q1 = np.where(shortest and dot < 0, -q1, q1)
    dot = np.where(shortest and dot < 0, -dot, dot)

    # Clamp to prevent instability at near 180° angle
    dot = np.clip(dot, -1, 1)

    # Compute the quaternion of the angle between the quaternions
    theta_0 = np.arccos(dot)  # theta_0 = angle between input vectors
    theta = theta_0 * t  # theta = angle between q0 vector and result

    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2 + 0.000001, axis=-1, keepdims=True)  # {q0, q2} is now an orthonormal basis

    return np.cos(theta) * q0 + np.sin(theta) * q2


def from_to(v1: np.ndarray, v2: np.ndarray, normalize_input: bool = True) -> np.ndarray:
    """
    Calculate the quaternion that rotates direction v1 to direction v2.
    When v1 and v2 are parallel, the result is the identity quaternion.

    Parameters
    ----------
    v1, v2 : np.array[..., [x,y,z]]
        Input vectors representing directions.
    normalize_input : bool
        Whether to normalize the input vectors.

    Returns
    -------
    rot : np.array[..., [w,x,y,z]]
        Quaternion representing the rotation.
    """
    assert v1.shape[-1] == 3 and v2.shape[-1] == 3, "Input vectors must have shape [..., 3]"
    assert v1.shape == v2.shape, "Input vectors must have the same shape"

    if normalize_input:
        v1_norm = vec.normalize(v1)
        v2_norm = vec.normalize(v2)
    else:
        v1_norm = v1
        v2_norm = v2

    if v1.ndim == 1:
        v1_norm = v1_norm[np.newaxis, :]
        v2_norm = v2_norm[np.newaxis, :]

    # Calculate cross product and dot product
    cross = np.cross(v1_norm, v2_norm)
    dot = np.sum(v1_norm * v2_norm, axis=-1, keepdims=True)

    # Handle general case
    axis_rot = normalize(cross)
    w = np.sqrt((1 + dot) * 0.5)  # cos(theta/2) = sqrt((1 + dot) / 2)
    s = np.sqrt((1 - dot) * 0.5)  # sin(theta/2) = sqrt((1 - dot) / 2)
    rot = np.concatenate([w, axis_rot * s], axis=-1)

    # Handle parallel vectors (dot ≈ 1)
    parallel = np.isclose(dot, 1.0)
    rot[parallel[..., 0]] = [1.0, 0.0, 0.0, 0.0]

    # Handle anti-parallel vectors (dot ≈ -1)
    anti_parallel_mask = np.isclose(dot, -1.0)[..., 0]

    if np.any(anti_parallel_mask):  # Check if any anti-parallel vectors exist
        v1_anti_parallel = v1_norm[anti_parallel_mask]  # Extract anti-parallel v1 vectors

        # Vectorized orthogonal vector selection
        orthogonal_anti_parallel = np.empty_like(v1_anti_parallel)  # Initialize with correct shape
        condition_mask = np.isclose(np.abs(v1_anti_parallel[..., 0]), 1.0)  # Boolean mask

        orthogonal_anti_parallel[condition_mask] = np.array([0.0, 1.0, 0.0])  # Assign for True condition
        orthogonal_anti_parallel[~condition_mask] = np.array([1.0, 0.0, 0.0])  # Assign for False condition

        # Vectorized axis of rotation calculation
        axis_rot_anti_parallel = normalize(np.cross(v1_anti_parallel, orthogonal_anti_parallel))

        # Vectorized quaternion construction for anti-parallel case
        rot_correction_anti_parallel = np.concatenate(
            [np.zeros_like(axis_rot_anti_parallel[..., :1]), axis_rot_anti_parallel], axis=-1
        )

        # Vectorized assignment of corrections
        rot[anti_parallel_mask] = rot_correction_anti_parallel

    if v1.ndim == 1:
        rot = rot[0]

    return rot


def from_to_axis(
    v1: np.ndarray, v2: np.ndarray, rot_axis: np.ndarray, normalize_input: bool = True
) -> np.ndarray:
    """
    Calculate the quaternion that rotates direction v1 to direction v2.
    The rotation axis is fixed to the provided axis.
    When v1 and v2 are parallel, the result is the identity quaternion.

    Parameters
    ----------
    v1, v2 : np.array[..., [x,y,z]]
        Input vectors representing directions.
    axis : np.array[..., [x,y,z]]
        Fixed rotation axis.
    normalize_input : bool
        Whether to normalize the input vectors.

    Returns
    -------
    rot : np.array[..., [w,x,y,z]]
        Quaternion representing the rotation.
    """
    assert v1.shape[-1] == 3 and v2.shape[-1] == 3, "Input vectors must have shape [..., 3]"
    assert v1.shape == v2.shape, "Input vectors must have the same shape"
    assert v1.shape == rot_axis.shape, "Input vectors and rotation axis must have the same shape"

    if rot_axis.ndim == 1:
        rot_axis = rot_axis[np.newaxis, :]

    if normalize_input:
        v1_norm = vec.normalize(v1)
        v2_norm = vec.normalize(v2)
    else:
        v1_norm = v1
        v2_norm = v2

    if v1.ndim == 1:
        v1_norm = v1_norm[np.newaxis, :]
        v2_norm = v2_norm[np.newaxis, :]

    # Calculate cross product and dot product
    cross = np.cross(v1_norm, v2_norm)
    dot = np.sum(v1_norm * v2_norm, axis=-1, keepdims=True)

    # Handle general case
    w = np.sqrt((1 + dot) * 0.5)  # cos(theta/2) = sqrt((1 + dot) / 2)
    s = np.sqrt((1 - dot) * 0.5)  # sin(theta/2) = sqrt((1 - dot) / 2)
    # Adjust sign of s based on cross product and rot_axis
    cross_dot_axis = np.sum(cross * rot_axis, axis=-1, keepdims=True)
    s *= np.sign(cross_dot_axis)  # Correct sign based on alignment
    # Combine w and s to form quaternion
    rot = np.concatenate([w, rot_axis * s], axis=-1)

    # Handle parallel vectors (dot ≈ 1)
    parallel = np.isclose(dot, 1.0)
    rot[parallel[..., 0]] = [1.0, 0.0, 0.0, 0.0]

    # Handle anti-parallel vectors (dot ≈ -1)
    anti_parallel = np.isclose(dot, -1.0)
    anti_parallel_rotmask = np.tile(anti_parallel, (1,) * (rot.ndim - 1) + (4,))
    anti_parallel_rotaxismask = np.tile(anti_parallel, (1,) * (rot_axis.ndim - 1) + (3,))
    rots_anti_parallel = rot[anti_parallel_rotmask]
    rots_anti_parallel[::4] = 0
    rots_anti_parallel[[False, True, True, True] * (len(rots_anti_parallel) // 4)] = rot_axis[
        anti_parallel_rotaxismask
    ]
    rot[anti_parallel_rotmask] = rots_anti_parallel

    if v1.ndim == 1:
        rot = rot[0]

    return rot


def _fast_cross(a: np.array, b: np.array) -> np.array:
    """
    Fast cross of two vectors

    Parameters
    ----------
    a : np.array[..., [x,y,z]]
    b : np.array[..., [x,y,z]]

    Returns
    -------
    np.array[..., [x,y,z]]
    """

    return np.concatenate(
        [
            a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
            a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
            a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1],
        ],
        axis=-1,
    )
