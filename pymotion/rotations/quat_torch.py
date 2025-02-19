from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
import pymotion.ops.vector_torch as vec


def from_scaled_angle_axis(scaledaxis: torch.Tensor) -> torch.Tensor:
    """
    Create a quaternion from an scaled angle-axis representation.

    Parameters
    ----------
    scaledaxis : torch.Tensor[..., [x,y,z]]
        axis [x,y,z] of rotation where magnitude is the angle of rotation

    Returns
    -------
    quat : torch.Tensor[..., [w,x,y,z]]
    """
    angle = torch.linalg.norm(scaledaxis, dim=-1).unsqueeze(-1)
    axis = scaledaxis / angle
    return from_angle_axis(angle, axis)


def from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Create a quaternion from an angle-axis representation.

    Parameters
    ----------
    angle : torch.Tensor[..., angle] in radians.
    axis : torch.Tensor[..., [x,y,z]]
        normalized axis [x,y,z] of rotation

    Returns
    -------
    quat : torch.Tensor[..., [w,x,y,z]]
    """
    c = torch.cos(angle / 2.0)
    s = torch.sin(angle / 2.0)
    return torch.cat((c, s * axis), dim=-1)


def from_euler(euler: torch.Tensor, order: np.array) -> torch.Tensor:
    """
    Create a quaternion from an euler representation with a specified order.

    Parameters
    ----------
    euler : torch.Tensor[..., [e0, e1, e2]]
        euler angles in radians
    order : np.array[..., ['x'|'y'|'z', 'x'|'y'|'z', 'x'|'y'|'z']]
        order of the euler angles
        symmetric orders not supported (e.g., XYX).

    Returns
    -------
    quat : torch.Tensor[..., [w,x,y,z]]
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
        torch.from_numpy(np.apply_along_axis(lambda x: axis[x.item()], -1, order[..., 0:1])).to(euler.device),
    )
    q1 = from_angle_axis(
        euler[..., 1:2],
        torch.from_numpy(np.apply_along_axis(lambda x: axis[x.item()], -1, order[..., 1:2])).to(euler.device),
    )
    q2 = from_angle_axis(
        euler[..., 2:3],
        torch.from_numpy(np.apply_along_axis(lambda x: axis[x.item()], -1, order[..., 2:3])).to(euler.device),
    )
    return mul(q0, mul(q1, q2))


def from_matrix(rotmats: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions.

    Parameters
    ----------
        rotmats: torch.Tensor[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                         [r1.x, r1.y, r1.z],
                                                         [r2.x, r2.y, r2.z]] where ri is row i.

    Returns
    -------
        quat torch.Tensor[..., [w,x,y,z]]
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
        torch.where(
            (r2c2 < 0.0).unsqueeze(-1),
            torch.where(
                (r0c0 > r1c1).unsqueeze(-1),
                torch.cat(
                    [
                        (r2c1 - r1c2).unsqueeze(-1),
                        (1.0 + r0c0 - r1c1 - r2c2).unsqueeze(-1),
                        (r1c0 + r0c1).unsqueeze(-1),
                        (r0c2 + r2c0).unsqueeze(-1),
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        (r0c2 - r2c0).unsqueeze(-1),
                        (r1c0 + r0c1).unsqueeze(-1),
                        (1.0 - r0c0 + r1c1 - r2c2).unsqueeze(-1),
                        (r2c1 + r1c2).unsqueeze(-1),
                    ],
                    dim=-1,
                ),
            ),
            torch.where(
                (r0c0 < -r1c1).unsqueeze(-1),
                torch.cat(
                    [
                        (r1c0 - r0c1).unsqueeze(-1),
                        (r0c2 + r2c0).unsqueeze(-1),
                        (r2c1 + r1c2).unsqueeze(-1),
                        (1.0 - r0c0 - r1c1 + r2c2).unsqueeze(-1),
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        (1.0 + r0c0 + r1c1 + r2c2).unsqueeze(-1),
                        (r2c1 - r1c2).unsqueeze(-1),
                        (r0c2 - r2c0).unsqueeze(-1),
                        (r1c0 - r0c1).unsqueeze(-1),
                    ],
                    dim=-1,
                ),
            ),
        )
    )


def to_euler(quaternions: torch.Tensor, order: np.array) -> torch.Tensor:
    """
    Convert a quaternion to an intrinsic euler representation with a specified order.
    Does not detect/solve gimbal lock.

    Parameters
    ----------
    quaternions : torch.Tensor[..., [w,x,y,z]]
    order : np.array[..., ['x'|'y'|'z', 'x'|'y'|'z', 'x'|'y'|'z']]
        order of the euler angles
        symmetric orders not supported (e.g., XYX).

    Returns
    -------
    euler : torch.Tensor[..., 3]
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

    i = (
        torch.from_numpy(np.apply_along_axis(lambda x: aux[x.item()], -1, order[..., 2:3])[..., np.newaxis])
        .to(quaternions.device)
        .type(torch.long)
    )
    j = (
        torch.from_numpy(np.apply_along_axis(lambda x: aux[x.item()], -1, order[..., 1:2])[..., np.newaxis])
        .to(quaternions.device)
        .type(torch.long)
    )
    k = (
        torch.from_numpy(np.apply_along_axis(lambda x: aux[x.item()], -1, order[..., 0:1])[..., np.newaxis])
        .to(quaternions.device)
        .type(torch.long)
    )

    # check if permutation is even or odd
    sign = (i - j) * (j - k) * (k - i) // 2

    # euler angles
    euler = torch.empty(quaternions.shape[:-1] + (3,), device=quaternions.device)

    # permutate quaternion elements
    a = quaternions[..., 0:1] - torch.take_along_dim(quaternions, j + 1, dim=-1)
    b = (
        torch.take_along_dim(quaternions, i + 1, dim=-1)
        + torch.take_along_dim(quaternions, k + 1, dim=-1) * sign
    )
    c = torch.take_along_dim(quaternions, j + 1, dim=-1) + quaternions[..., 0:1]
    d = torch.take_along_dim(quaternions, k + 1, dim=-1) * sign - torch.take_along_dim(
        quaternions, i + 1, dim=-1
    )

    # compute second angle
    euler[..., 1:2] = (2 * torch.arctan2(torch.hypot(c, d), torch.hypot(a, b))) - (torch.pi / 2)

    # compute first and third angle
    half_sum = torch.arctan2(b, a)
    half_diff = torch.arctan2(d, c)
    euler[..., angle_first : angle_first + 1] = half_sum - half_diff
    euler[..., angle_third : angle_third + 1] = (half_sum + half_diff) * sign

    # for i in range(3):
    # if euler[..., i] < -np.pi:
    #    euler[..., i] += 2 * np.pi
    # elif euler[..., i] > np.pi:
    #    euler[..., i] -= 2 * np.pi
    euler = torch.remainder(euler, 2 * torch.pi)

    return euler


def to_scaled_angle_axis(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion to scaled axis angle representation.

    Parameters
    ----------
    quaternions : torch.Tensor[..., [w,x,y,z]]

    Returns
    -------
    scaledaxis : torch.Tensor[..., [x,y,z]]
        axis [x,y,z] of rotation where magnitude is the angle of rotation
    """
    angle, axis = to_angle_axis(quaternions)
    return angle * axis


def to_angle_axis(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion to scaled axis angle representation.

    Parameters
    ----------
    quaternions : torch.Tensor[..., [w,x,y,z]]

    Returns
    -------
    angle: torch.Tensor[..., angle]
    axis : torch.Tensor[..., [x,y,z]]
        normalized axis [x,y,z] of rotation
    """
    w = quaternions[..., 0]
    xyz = quaternions[..., 1:]

    angle = 2 * torch.arccos(torch.clip(w, -1.0, 1.0))
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

    # Avoid division by zero when s is close to zero (identity quaternion)
    axis = torch.zeros_like(xyz)
    mask = s > 1e-8
    if mask.any():
        axis[mask] = xyz[mask] / s[mask].unsqueeze(-1)

    return angle.unsqueeze(-1), axis


def to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Parameters
    ----------
        quaternions: torch.Tensor[..., [w,x,y,z]]

    Returns
    -------
        rotmats: torch.Tensor[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                         [r1.x, r1.y, r1.z],
                                                         [r2.x, r2.y, r2.z]] where ri is row i.
    """
    qw, qx, qy, qz = torch.unbind(quaternions, -1)

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

    m = torch.empty(quaternions.shape[:-1] + (3, 3), device=quaternions.device)
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


def mul_vec(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Multiply a vector by a quaternion

    Parameters
    ----------
    q : torch.Tensor[..., [w,x,y,z]]
    v : torch.Tensor[..., [x,y,z]]

    Returns
    -------
    v: torch.Tensor[..., [x,y,z]]
    """
    t = 2.0 * _fast_cross(q[..., 1:], v)
    return v + q[..., 0].unsqueeze(-1) * t + _fast_cross(q[..., 1:], t)


def mul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.

    Parameters
    ----------
    q0 : torch.Tensor[..., [w,x,y,z]]
    q1 : torch.Tensor[..., [w,x,y,z]]

    Returns
    -------
    quat : torch.Tensor[..., [w,x,y,z]]
    """

    q0 = q0.clone()
    q1 = q1.clone()
    w0, x0, y0, z0 = q0[..., 0:1], q0[..., 1:2], q0[..., 2:3], q0[..., 3:4]
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    # (w0,v0)(w1,v1) = (w0w1 - v0·v1, w0v1 + w1v0 + v0 x v1)
    return torch.cat(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,  # w
            w0 * x1 + w1 * x0 + y0 * z1 - z0 * y1,  # x
            w0 * y1 + w1 * y0 + z0 * x1 - x0 * z1,  # y
            w0 * z1 + w1 * z0 + x0 * y1 - y0 * x1,  # z
        ),
        dim=-1,
    )


def length(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Get the length or magnitude of the quaternions.

    Parameters
    ----------
    quaternions : torch.Tensor[..., [w,x,y,z]]

    Returns
    -------
    length : torch.Tensor[...]
    """
    return torch.linalg.norm(quaternions, dim=-1)


def inverse(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Inverse of a quaternion.

    Parameters
    ----------
    quaternions : torch.Tensor[..., [w,x,y,z]]

    Returns
    -------
    quaternions : torch.Tensor[..., [w,x,y,z]]
    """
    # for a unit quaternion the conjugate is the inverse
    # q^-1 = [q0, -q1, -q2, -q3]
    return conjugate(quaternions)


def conjugate(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Compute the conjugate of a quaternion.

    Parameters
    ----------
    quaternions : torch.Tensor[..., [w,x,y,z]]

    Returns
    -------
    quaternions : torch.Tensor[..., [w,x,y,z]]
    """
    return torch.cat((quaternions[..., 0:1], -quaternions[..., 1:]), dim=-1)


def normalize(quaternions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert all quaternions to unit quatenrions.

    Parameters
    ----------
    quaternions : torch.Tensor[..., [w,x,y,z]]

    Returns
    -------
    quaternions : torch.Tensor[..., [w,x,y,z]]
    """
    return quaternions / (length(quaternions).unsqueeze(-1) + eps)


def unroll(quaternions: torch.Tensor, dim: int) -> torch.Tensor:
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
    quaternions : torch.Tensor[..., [w,x,y,z]]
    dim : int
        unroll dimension (e.g., frames dimension)

    Returns
    -------
    quaternions : torch.Tensor[..., [w,x,y,z]]
    """
    r = quaternions.swapaxes(0, dim)
    # start with the second quaternion since
    # we keep the cover of the first one
    for i in range(1, r.shape[0]):
        # distance (dot product) between the previous and current quaternion
        d0 = torch.sum(r[i] * r[i - 1], dim=-1)
        # distance (dot product) between the previous and flipped current quaternion
        d1 = torch.sum(-r[i] * r[i - 1], dim=-1)
        # if the distance with the flipped quaternion is smaller, use it
        r[i][d0 < d1] = -r[i][d0 < d1]
    r = r.swapaxes(0, dim)
    return r


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: float | torch.Tensor, shortest: bool = True) -> torch.Tensor:
    """
    Perform spherical linear interpolation (SLERP) between two unit quaternions.

    Parameters
    ----------
    q0 : torch.Tensor[..., [w,x,y,z]]
    q1 : torch.Tensor[..., [w,x,y,z]]
    t : float or torch.Tensor[..., [t]]
        Interpolation parameter between 0 and 1. At t=0, returns q0 and at t=1, returns q1.
    shorthest : bool
        Ensure the shorthest path between quaternions.

    Returns
    -------
    quat : torch.Tensor[..., [w,x,y,z]]
    """

    # Compute the cosine of the angle between the two vectors.
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    # If the dot product is negative, the quaternions
    # have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if shortest:
        q1 = torch.where(dot < 0, -q1, q1)
        dot = torch.where(dot < 0, -dot, dot)

    # Clamp to prevent instability at near 180° angle
    dot = torch.clip(dot, -1, 1)

    # Compute the quaternion of the angle between the quaternions
    theta_0 = torch.arccos(dot)  # theta_0 = angle between input vectors
    theta = theta_0 * t  # theta = angle between q0 vector and result

    q2 = q1 - q0 * dot
    q2 /= torch.linalg.norm(q2 + 0.000001, dim=-1, keepdim=True)  # {q0, q2} is now an orthonormal basis

    return torch.cos(theta) * q0 + torch.sin(theta) * q2


def from_to(v1: torch.Tensor, v2: torch.Tensor, normalize_input: bool = True) -> torch.Tensor:
    """
    Calculate the quaternion that rotates direction v1 to direction v2 using PyTorch.
    When v1 and v2 are parallel, the result is the identity quaternion.

    Parameters
    ----------
    v1, v2 : torch.Tensor[..., [x,y,z]]
        Input vectors representing directions.
    normalize_input : bool
        Whether to normalize the input vectors.

    Returns
    -------
    rot : torch.Tensor[..., [w,x,y,z]]
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
        v1_norm = v1_norm.unsqueeze(0)
        v2_norm = v2_norm.unsqueeze(0)

    # Calculate cross product and dot product
    cross = torch.linalg.cross(v1_norm, v2_norm)
    dot = torch.sum(v1_norm * v2_norm, dim=-1, keepdim=True)

    # Handle general case
    axis_rot = normalize(cross)
    w = torch.sqrt((1 + dot) * 0.5)  # cos(theta/2) = sqrt((1 + dot) / 2)
    s = torch.sqrt((1 - dot) * 0.5)  # sin(theta/2) = sqrt((1 - dot) / 2)
    rot = torch.cat([w, axis_rot * s], dim=-1)

    # Handle parallel vectors (dot ≈ 1)
    parallel_mask = torch.isclose(dot, torch.tensor(1.0, device=rot.device, dtype=rot.dtype))
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=rot.device, dtype=rot.dtype)
    rot[parallel_mask[..., 0]] = identity_quat

    # Handle anti-parallel vectors (dot ≈ -1)
    anti_parallel_mask = torch.isclose(dot, torch.tensor(-1.0, device=rot.device, dtype=rot.dtype))[..., 0]

    if torch.any(anti_parallel_mask):  # Check if any anti-parallel vectors exist
        v1_anti_parallel = v1_norm[anti_parallel_mask]  # Extract anti-parallel v1 vectors

        # Vectorized orthogonal vector selection
        orthogonal_anti_parallel = torch.empty_like(v1_anti_parallel)  # Initialize with correct shape
        condition_mask = torch.isclose(
            torch.abs(v1_anti_parallel[..., 0]), torch.tensor(1.0, device=rot.device, dtype=rot.dtype)
        )  # Boolean mask

        orthogonal_anti_parallel[condition_mask] = torch.tensor(
            [0.0, 1.0, 0.0], device=rot.device, dtype=rot.dtype
        )  # Assign for True condition
        orthogonal_anti_parallel[~condition_mask] = torch.tensor(
            [1.0, 0.0, 0.0], device=rot.device, dtype=rot.dtype
        )  # Assign for False condition

        # Vectorized axis of rotation calculation
        axis_rot_anti_parallel = normalize(torch.linalg.cross(v1_anti_parallel, orthogonal_anti_parallel))

        # Vectorized quaternion construction for anti-parallel case
        rot_correction_anti_parallel = torch.cat(
            [torch.zeros_like(axis_rot_anti_parallel[..., :1]), axis_rot_anti_parallel], dim=-1
        )

        # Vectorized assignment of corrections
        rot[anti_parallel_mask] = rot_correction_anti_parallel

    if v1.ndim == 1:
        rot = rot.squeeze(0)

    return rot


def from_to_axis(
    v1: torch.Tensor, v2: torch.Tensor, rot_axis: torch.Tensor, normalize_input: bool = True
) -> torch.Tensor:
    """
    Calculate the quaternion that rotates direction v1 to direction v2.
    The rotation axis is fixed to the provided axis.
    When v1 and v2 are parallel, the result is the identity quaternion.

    Parameters
    ----------
    v1, v2 : torch.Tensor[..., [x,y,z]]
        Input vectors representing directions.
    rot_axis : torch.Tensor[..., [x,y,z]]
        Fixed rotation axis.
    normalize_input : bool
        Whether to normalize the input vectors.

    Returns
    -------
    rot : torch.Tensor[..., [w,x,y,z]]
        Quaternion representing the rotation.
    """
    assert v1.shape[-1] == 3 and v2.shape[-1] == 3, "Input vectors must have shape [..., 3]"
    assert v1.shape == v2.shape, "Input vectors must have the same shape"
    assert v1.shape == rot_axis.shape, "Input vectors must have the same shape"

    if rot_axis.ndim == 1:
        rot_axis = rot_axis.unsqueeze(0)

    if normalize_input:
        v1_norm = vec.normalize(v1)
        v2_norm = vec.normalize(v2)
    else:
        v1_norm = v1
        v2_norm = v2

    if v1.ndim == 1:
        v1_norm = v1_norm.unsqueeze(0)
        v2_norm = v2_norm.unsqueeze(0)

    # Calculate cross product and dot product
    cross = torch.cross(v1_norm, v2_norm, dim=-1)
    dot = torch.sum(v1_norm * v2_norm, dim=-1, keepdim=True)

    # Handle general case
    w = torch.sqrt((1 + dot) * 0.5)  # cos(theta/2) = sqrt((1 + dot) / 2)
    s = torch.sqrt((1 - dot) * 0.5)  # sin(theta/2) = sqrt((1 - dot) / 2)
    # Adjust sign of s based on cross product and rot_axis
    cross_dot_axis = torch.sum(cross * rot_axis, dim=-1, keepdim=True)
    s *= torch.sign(cross_dot_axis)  # Correct sign based on alignment
    # Combine w and s to form quaternion
    rot = torch.cat([w, rot_axis * s], dim=-1)

    # Handle parallel vectors (dot ≈ 1)
    parallel_mask = torch.isclose(dot, torch.tensor(1.0, device=rot.device, dtype=rot.dtype))
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=rot.device, dtype=rot.dtype)
    rot[parallel_mask[..., 0]] = identity_quat

    # Handle anti-parallel vectors (dot ≈ -1)
    anti_parallel = torch.isclose(dot, torch.tensor(-1.0, device=rot.device, dtype=rot.dtype))
    anti_parallel_rotmask = torch.tile(anti_parallel, (1,) * (rot.ndim - 1) + (4,))
    anti_parallel_rotaxismask = torch.tile(anti_parallel, (1,) * (rot_axis.ndim - 1) + (3,))
    rots_anti_parallel = rot[anti_parallel_rotmask]
    rots_anti_parallel[::4] = 0
    rots_anti_parallel[[False, True, True, True] * (len(rots_anti_parallel) // 4)] = rot_axis[
        anti_parallel_rotaxismask
    ]
    rot[anti_parallel_rotmask] = rots_anti_parallel

    if v1.ndim == 1:
        rot = rot.squeeze(0)

    return rot


def _fast_cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Fast cross of two vectors

    Parameters
    ----------
    a : torch.Tensor[..., [x,y,z]]
    b : torch.Tensor[..., [x,y,z]]

    Returns
    -------
    torch.Tensor[..., [x,y,z]]
    """

    return torch.cat(
        [
            a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
            a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
            a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1],
        ],
        dim=-1,
    )
