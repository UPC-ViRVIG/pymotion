# Portions Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unified dual quaternion operations wrapper that dispatches to NumPy or PyTorch backends
based on input tensor types. This provides a single API for dual quaternion operations
regardless of the underlying array library being used.

Dual quaternions are represented as arrays of shape [..., 8]
where the last dimension is the dual quaternion representation.
The first 4 elements are the real part and the last 4 elements are the dual part.
[..., [w_r, x_r, y_r, z_r, w_d, x_d, y_d, z_d]]

The dispatcher imports backends at module load time. If a library isn't installed,
its backend will be None and will raise an error at runtime if you try to use it.
"""
from __future__ import annotations


# Import type references and backend modules at module import time
_TorchTensor = None
_dual_quat_torch = None

try:
    import torch

    _TorchTensor = torch.Tensor
    from . import dual_quat_torch as _dual_quat_torch
except ImportError:
    pass

_NumpyArray = None
_dual_quat_np = None

try:
    import numpy as np

    _NumpyArray = np.ndarray
    from . import dual_quat_np as _dual_quat_np
except ImportError:
    pass


# Explicit wrapper functions with full documentation
def from_rotation_translation(rotations, translations):
    """
    Convert the rotations (quaternions) and translation (3D vectors) information to dual quaternions.

    Parameters
    ----------
    rotations : torch.Tensor or np.array[..., [w, x, y, z]]
        Rotation quaternions
    translations : torch.Tensor or np.array[..., 3]
        Translation vectors

    Returns
    -------
    dq : torch.Tensor or np.array[..., 8]
        Dual quaternions
    """
    if _TorchTensor is not None and isinstance(rotations, _TorchTensor):
        return _dual_quat_torch.from_rotation_translation(rotations, translations)
    else:
        return _dual_quat_np.from_rotation_translation(rotations, translations)


def from_translation(translations):
    """
    Convert a translation to a dual quaternion.

    Parameters
    ----------
    translations : torch.Tensor or np.array[..., 3]
        Translation vectors

    Returns
    -------
    dual_quats : torch.Tensor or np.array[..., 8]
        Dual quaternions
    """
    if _TorchTensor is not None and isinstance(translations, _TorchTensor):
        return _dual_quat_torch.from_translation(translations)
    else:
        return _dual_quat_np.from_translation(translations)


def to_rotation_translation(dq):
    """
    Convert a dual quaternion to the rotations (quaternions) and translations (3D vectors).

    Parameters
    ----------
    dq : torch.Tensor or np.array[..., 8]
        Dual quaternions

    Returns
    -------
    rotations : torch.Tensor or np.array[..., [w, x, y, z]]
        Rotation quaternions
    translations : torch.Tensor or np.array[..., 3]
        Translation vectors
    """
    if _TorchTensor is not None and isinstance(dq, _TorchTensor):
        return _dual_quat_torch.to_rotation_translation(dq)
    else:
        return _dual_quat_np.to_rotation_translation(dq)


def normalize(dq):
    """
    Normalize the dual quaternion to unit length and make sure that
    the dual part is orthogonal to the real part (unit dual quaternion).

    Parameters
    ----------
    dq : torch.Tensor or np.array[..., 8]
        Dual quaternions

    Returns
    -------
    dq : torch.Tensor or np.array[..., 8]
        Normalized dual quaternions
    """
    if _TorchTensor is not None and isinstance(dq, _TorchTensor):
        return _dual_quat_torch.normalize(dq)
    else:
        return _dual_quat_np.normalize(dq)


def is_unit(dq, atol=1e-03):
    """
    Check if the dual quaternion is a unit one.
    A unit dual quaternion satisfies two properties:
    - The norm of the real part is 1
    - The dot product of the real and dual part is 0.

    Parameters
    ----------
    dq : torch.Tensor or np.array[..., 8]
        Dual quaternions
    atol : float, optional
        Absolute tolerance for the check. Default is 1e-03.

    Returns
    -------
    is_unit : bool
        True if the dual quaternion is a unit one
    """
    if _TorchTensor is not None and isinstance(dq, _TorchTensor):
        return _dual_quat_torch.is_unit(dq, atol)
    else:
        return _dual_quat_np.is_unit(dq, atol)


def unroll(dq, axis):
    """
    Enforce dual quaternion continuity across the time dimension by selecting
    the representation (dq or -dq) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Parameters
    ----------
    dq : torch.Tensor or np.array[..., 8]
        Dual quaternions
    axis : int
        Unroll axis (e.g., frames axis)

    Returns
    -------
    dq : torch.Tensor or np.array[..., 8]
        Unrolled dual quaternions
    """
    if _TorchTensor is not None and isinstance(dq, _TorchTensor):
        return _dual_quat_torch.unroll(dq, axis)
    else:
        return _dual_quat_np.unroll(dq, axis)


# Expose public API
__all__ = [
    "from_rotation_translation",
    "from_translation",
    "to_rotation_translation",
    "normalize",
    "is_unit",
    "unroll",
]
