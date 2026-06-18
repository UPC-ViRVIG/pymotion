# Portions Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unified quaternion operations wrapper that dispatches to NumPy or PyTorch backends
based on input tensor types. This provides a single API for quaternion operations
regardless of the underlying array library being used.

The dispatcher imports backends at module load time. If a library isn't installed,
its backend will be None and will raise an error at runtime if you try to use it.
"""

from __future__ import annotations

# Import type references and backend modules at module import time
_TorchTensor = None
_quat_torch = None

try:
    import torch

    _TorchTensor = torch.Tensor
    from . import quat_torch as _quat_torch
except ImportError:
    pass

_NumpyArray = None
_quat_np = None

try:
    import numpy as np

    _NumpyArray = np.ndarray
    from . import quat_np as _quat_np
except ImportError:
    pass


# Explicit wrapper functions
def from_scaled_angle_axis(scaledaxis):
    """
    Create a quaternion from an scaled angle-axis representation.

    Parameters
    ----------
    scaledaxis : torch.Tensor or np.array[..., [x,y,z]]
        axis [x,y,z] of rotation where magnitude is the angle of rotation

    Returns
    -------
    quat : torch.Tensor or np.array[..., [w,x,y,z]]
    """
    if _TorchTensor is not None and isinstance(scaledaxis, _TorchTensor):
        return _quat_torch.from_scaled_angle_axis(scaledaxis)
    else:
        return _quat_np.from_scaled_angle_axis(scaledaxis)


def from_angle_axis(angle, axis):
    """
    Create a quaternion from an angle-axis representation.

    Parameters
    ----------
    angle : torch.Tensor or np.array[..., angle] in radians.
    axis : torch.Tensor or np.array[..., [x,y,z]]
        normalized axis [x,y,z] of rotation

    Returns
    -------
    quat : torch.Tensor or np.array[..., [w,x,y,z]]
    """
    if _TorchTensor is not None and isinstance(angle, _TorchTensor):
        return _quat_torch.from_angle_axis(angle, axis)
    else:
        return _quat_np.from_angle_axis(angle, axis)


def from_euler(euler, order):
    """
    Create a quaternion from an euler representation with a specified order.

    Parameters
    ----------
    euler : torch.Tensor or np.array[..., [e0, e1, e2]]
        euler angles in radians
    order : np.array[..., ['x'|'y'|'z', 'x'|'y'|'z', 'x'|'y'|'z']]
        order of the euler angles
        symmetric orders not supported (e.g., XYX).

    Returns
    -------
    quat : torch.Tensor or np.array[..., [w,x,y,z]]
    """
    if _TorchTensor is not None and isinstance(euler, _TorchTensor):
        return _quat_torch.from_euler(euler, order)
    else:
        return _quat_np.from_euler(euler, order)


def from_matrix(rotmats):
    """
    Convert rotation matrices to quaternions.

    Parameters
    ----------
    rotmats: torch.Tensor or np.array[..., 3, 3]
        Matrix order: [[r0.x, r0.y, r0.z],
                       [r1.x, r1.y, r1.z],
                       [r2.x, r2.y, r2.z]] where ri is row i.

    Returns
    -------
    quat : torch.Tensor or np.array[..., [w,x,y,z]]
    """
    if _TorchTensor is not None and isinstance(rotmats, _TorchTensor):
        return _quat_torch.from_matrix(rotmats)
    else:
        return _quat_np.from_matrix(rotmats)


def to_euler(quaternions, order):
    """
    Convert a quaternion to an intrinsic euler representation with a specified order.
    Does not detect/solve gimbal lock.

    Parameters
    ----------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]
    order : np.array[..., ['x'|'y'|'z', 'x'|'y'|'z', 'x'|'y'|'z']]
        order of the euler angles
        symmetric orders not supported (e.g., XYX).

    Returns
    -------
    euler : torch.Tensor or np.array[..., 3]
        euler angles in radians
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.to_euler(quaternions, order)
    else:
        return _quat_np.to_euler(quaternions, order)


def to_scaled_angle_axis(quaternions):
    """
    Quaternion to scaled axis angle representation.

    Parameters
    ----------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]

    Returns
    -------
    scaledaxis : torch.Tensor or np.array[..., [x,y,z]]
        axis [x,y,z] of rotation where magnitude is the angle of rotation
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.to_scaled_angle_axis(quaternions)
    else:
        return _quat_np.to_scaled_angle_axis(quaternions)


def to_angle_axis(quaternions):
    """
    Quaternion to axis angle representation.

    Parameters
    ----------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]

    Returns
    -------
    angle: torch.Tensor or np.array[..., angle]
    axis : torch.Tensor or np.array[..., [x,y,z]]
        normalized axis [x,y,z] of rotation
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.to_angle_axis(quaternions)
    else:
        return _quat_np.to_angle_axis(quaternions)


def to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Parameters
    ----------
    quaternions: torch.Tensor or np.array[..., [w,x,y,z]]

    Returns
    -------
    rotmats: torch.Tensor or np.array[..., 3, 3]
        Matrix order: [[r0.x, r0.y, r0.z],
                       [r1.x, r1.y, r1.z],
                       [r2.x, r2.y, r2.z]] where ri is row i.
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.to_matrix(quaternions)
    else:
        return _quat_np.to_matrix(quaternions)


def mul_vec(q, v):
    """
    Multiply a vector by a quaternion.

    Parameters
    ----------
    q : torch.Tensor or np.array[..., [w,x,y,z]]
    v : torch.Tensor or np.array[..., [x,y,z]]

    Returns
    -------
    v: torch.Tensor or np.array[..., [x,y,z]]
    """
    if _TorchTensor is not None and isinstance(q, _TorchTensor):
        return _quat_torch.mul_vec(q, v)
    else:
        return _quat_np.mul_vec(q, v)


def mul(q0, q1):
    """
    Multiply two quaternions.

    Parameters
    ----------
    q0 : torch.Tensor or np.array[..., [w,x,y,z]]
    q1 : torch.Tensor or np.array[..., [w,x,y,z]]

    Returns
    -------
    quat : torch.Tensor or np.array[..., [w,x,y,z]]
    """
    if _TorchTensor is not None and isinstance(q0, _TorchTensor):
        return _quat_torch.mul(q0, q1)
    else:
        return _quat_np.mul(q0, q1)


def length(quaternions):
    """
    Get the length or magnitude of the quaternions.

    Parameters
    ----------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]

    Returns
    -------
    length : torch.Tensor or np.array[...]
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.length(quaternions)
    else:
        return _quat_np.length(quaternions)


def inverse(quaternions):
    """
    Inverse of a quaternion.

    Parameters
    ----------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.inverse(quaternions)
    else:
        return _quat_np.inverse(quaternions)


def conjugate(quaternions):
    """
    Compute the conjugate of a quaternion.

    Parameters
    ----------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.conjugate(quaternions)
    else:
        return _quat_np.conjugate(quaternions)


def normalize(quaternions, eps=1e-8):
    """
    Convert all quaternions to unit quaternions.

    Parameters
    ----------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-8.

    Returns
    -------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.normalize(quaternions, eps)
    else:
        return _quat_np.normalize(quaternions, eps)


def unroll(quaternions, dim):
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
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]
    dim : int
        unroll dimension (e.g., frames dimension)

    Returns
    -------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.unroll(quaternions, dim)
    else:
        return _quat_np.unroll(quaternions, dim)


def slerp(q0, q1, t, shortest=True):
    """
    Perform spherical linear interpolation (SLERP) between two unit quaternions.

    Parameters
    ----------
    q0 : torch.Tensor or np.array[..., [w,x,y,z]]
    q1 : torch.Tensor or np.array[..., [w,x,y,z]]
    t : float or torch.Tensor or np.array[..., [t]]
        Interpolation parameter between 0 and 1. At t=0, returns q0 and at t=1, returns q1.
    shortest : bool, optional
        Ensure the shortest path between quaternions. Default is True.

    Returns
    -------
    quat : torch.Tensor or np.array[..., [w,x,y,z]]
    """
    if _TorchTensor is not None and isinstance(q0, _TorchTensor):
        return _quat_torch.slerp(q0, q1, t, shortest)
    else:
        return _quat_np.slerp(q0, q1, t, shortest)


def weighted_slerp(quaternions, weights, shortest=True):
    """
    Perform weighted spherical linear interpolation (SLERP) across multiple quaternions.
    Uses iterative weighted SLERP to properly blend quaternions based on their weights.

    Parameters
    ----------
    quaternions : torch.Tensor or np.array[num_items, ..., [w,x,y,z]]
        Quaternions to blend. First dimension is the items to blend.
        For batch processing with different weights, call this function in a loop.
    weights : torch.Tensor or np.array[num_items]
        Weights for each quaternion. Must sum to approximately 1.0 for meaningful results.
    shortest : bool, optional
        Ensure the shortest path between quaternions. Default is True.

    Returns
    -------
    quat : torch.Tensor or np.array[..., [w,x,y,z]]
        Weighted blend of input quaternions

    Notes
    -----
    This function performs iterative weighted SLERP:
    1. Start with first quaternion
    2. For each subsequent quaternion i:
       - Compute t_i = weight_i / (accumulated_weight + weight_i)
       - SLERP between accumulated result and quaternion i with parameter t_i
       - Update accumulated_weight += weight_i

    This approach ensures proper geodesic blending on the quaternion manifold.

    Examples
    --------
    >>> import numpy as np
    >>> import pymotion.rotations.quat as quat
    >>>
    >>> # Define 3 quaternions (identity, 90° around Z, 180° around Z)
    >>> quaternions = np.array([
    ...     [1.0, 0.0, 0.0, 0.0],  # identity
    ...     [0.707, 0.0, 0.0, 0.707],  # 90° Z
    ...     [0.0, 0.0, 0.0, 1.0],  # 180° Z
    ... ])
    >>>
    >>> # Single weight vector
    >>> weights = np.array([0.33, 0.33, 0.34])
    >>> result = quat.weighted_slerp(quaternions, weights)
    >>>
    >>> # Batch processing with different weights - call in a loop
    >>> weights_batch = np.array([
    ...     [1.0, 0.0, 0.0],  # Only first quaternion
    ...     [0.0, 1.0, 0.0],  # Only second quaternion
    ...     [0.5, 0.5, 0.0],  # Blend first two
    ... ])
    >>> results = np.stack([
    ...     quat.weighted_slerp(quaternions, w) for w in weights_batch
    ... ])
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.weighted_slerp(quaternions, weights, shortest)
    else:
        return _quat_np.weighted_slerp(quaternions, weights, shortest)


def from_to(v1, v2, normalize_input=True):
    """
    Calculate the quaternion that rotates direction v1 to direction v2.
    When v1 and v2 are parallel, the result is the identity quaternion.

    Parameters
    ----------
    v1, v2 : torch.Tensor or np.array[..., [x,y,z]]
        Input vectors representing directions.
    normalize_input : bool, optional
        Whether to normalize the input vectors. Default is True.

    Returns
    -------
    rot : torch.Tensor or np.array[..., [w,x,y,z]]
        Quaternion representing the rotation.
    """
    if _TorchTensor is not None and isinstance(v1, _TorchTensor):
        return _quat_torch.from_to(v1, v2, normalize_input)
    else:
        return _quat_np.from_to(v1, v2, normalize_input)


def from_to_axis(v1, v2, rot_axis, normalize_input=True):
    """
    Calculate the quaternion that rotates direction v1 to direction v2.
    The rotation axis is fixed to the provided axis.
    When v1 and v2 are parallel, the result is the identity quaternion.

    Parameters
    ----------
    v1, v2 : torch.Tensor or np.array[..., [x,y,z]]
        Input vectors representing directions.
    rot_axis : torch.Tensor or np.array[..., [x,y,z]]
        Fixed rotation axis.
    normalize_input : bool, optional
        Whether to normalize the input vectors. Default is True.

    Returns
    -------
    rot : torch.Tensor or np.array[..., [w,x,y,z]]
        Quaternion representing the rotation.
    """
    if _TorchTensor is not None and isinstance(v1, _TorchTensor):
        return _quat_torch.from_to_axis(v1, v2, rot_axis, normalize_input)
    else:
        return _quat_np.from_to_axis(v1, v2, rot_axis, normalize_input)


def canonicalize(quaternions):
    """
    Convert quaternions to canonical form where w >= 0.

    This resolves the quaternion double-cover problem by ensuring all quaternions
    use the same hemisphere. This is important before operations like scaling
    the rotation angle, as it ensures the shortest rotation path is used.

    The PyTorch implementation is fully differentiable.

    Parameters
    ----------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]
        Quaternions with w >= 0 (same rotation, canonical representation)

    Notes
    -----
    A quaternion q and -q represent the same rotation. However, when converting
    to angle-axis representation, a quaternion with w < 0 will produce an angle
    > 180°, which represents the "long way around". By ensuring w >= 0, we
    guarantee the angle will be <= 180° (the shortest rotation path).

    Examples
    --------
    >>> q = np.array([[-0.707, 0.0, 0.707, 0.0]])  # w < 0
    >>> q_canonical = canonicalize(q)
    >>> print(q_canonical)  # [0.707, 0.0, -0.707, 0.0], same rotation but w >= 0
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.canonicalize(quaternions)
    else:
        return _quat_np.canonicalize(quaternions)


def scale(quaternions, factor):
    """
    Scale quaternion rotations by a factor.

    This function scales the rotation angle of quaternions while preserving
    the rotation axis. A factor of 0.5 gives half the rotation, 2.0 doubles it, etc.

    The function automatically handles the quaternion double-cover problem by
    canonicalizing quaternions before scaling, ensuring the shortest rotation
    path is always used.

    The PyTorch implementation is fully differentiable.

    Parameters
    ----------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]
        Input quaternions to scale.
    factor : float, torch.Tensor, or np.array
        Scale factor for the rotation angle. Can be:
        - A scalar (applies same scale to all quaternions)
        - A tensor/array broadcastable to quaternions shape (for per-element scaling)

    Returns
    -------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]
        Scaled quaternions.

    Notes
    -----
    The scaling is performed in angle-axis space:
    1. Canonicalize quaternions (ensure w >= 0 for shortest path)
    2. Convert to scaled angle-axis representation
    3. Multiply the angle by the scale factor
    4. Convert back to quaternion

    Examples
    --------
    >>> q = np.array([[0.707, 0.0, 0.707, 0.0]])  # 90° rotation around Y
    >>> q_half = scale(q, 0.5)  # 45° rotation around Y
    >>> q_double = scale(q, 2.0)  # 180° rotation around Y

    >>> # Per-joint scaling
    >>> q = np.random.randn(10, 22, 4)  # 10 frames, 22 joints
    >>> q = normalize(q)
    >>> joint_scales = np.random.rand(22, 1)  # Different scale per joint
    >>> q_scaled = scale(q, joint_scales)
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _quat_torch.scale(quaternions, factor)
    else:
        return _quat_np.scale(quaternions, factor)



def delta(rotations, target_rotations):
    """
    Compute the shortest-path delta rotation from rotations to target_rotations.

    For each pair (q, q_target), computes q_delta = q^{-1} * q_target such that
    q * q_delta = q_target.  Before multiplying, the target quaternion is flipped
    to the same hemisphere as the source quaternion (via the dot-product sign),
    which guarantees the delta always represents the shortest angular path.

    The PyTorch implementation is fully differentiable.

    Parameters
    ----------
    rotations : torch.Tensor or np.array[..., [w,x,y,z]]
        Source quaternions.
    target_rotations : torch.Tensor or np.array[..., [w,x,y,z]]
        Target quaternions.

    Returns
    -------
    torch.Tensor or np.array[..., [w,x,y,z]]
        Delta quaternions such that ``mul(rotations, delta) ≈ target_rotations``
        via the shortest path.

    Examples
    --------
    >>> q0 = np.array([[1.0, 0.0, 0.0, 0.0]])           # identity
    >>> q1 = np.array([[0.707, 0.0, 0.707, 0.0]])        # 90° around Y
    >>> d  = delta(q0, q1)
    >>> # mul(q0, d) ≈ q1
    """
    if _TorchTensor is not None and isinstance(rotations, _TorchTensor):
        return _quat_torch.delta(rotations, target_rotations)
    else:
        return _quat_np.delta(rotations, target_rotations)


# Expose public API
__all__ = [
    "from_scaled_angle_axis",
    "from_angle_axis",
    "from_euler",
    "from_matrix",
    "to_euler",
    "to_scaled_angle_axis",
    "to_angle_axis",
    "to_matrix",
    "mul_vec",
    "mul",
    "length",
    "inverse",
    "conjugate",
    "normalize",
    "unroll",
    "slerp",
    "weighted_slerp",
    "from_to",
    "from_to_axis",
    "canonicalize",
    "scale",
    "delta",
]
