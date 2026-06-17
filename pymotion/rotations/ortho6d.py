# Portions Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unified ortho6D operations wrapper that dispatches to NumPy or PyTorch backends
based on input tensor types. This provides a single API for ortho6D operations
regardless of the underlying array library being used.

Ortho6D are represented as tensors of shape [..., 3, 2]
where the last dimension is the ortho6D representations,
which is the first two columns of the rotation matrix.
Matrix order: [[r0.x, r0.y],
               [r1.x, r1.y],
               [r2.x, r2.y]] where ri is row i.

The dispatcher imports backends at module load time. If a library isn't installed,
its backend will be None and will raise an error at runtime if you try to use it.
"""

from __future__ import annotations

# Import type references and backend modules at module import time
_TorchTensor = None
_ortho6d_torch = None

try:
    import torch

    _TorchTensor = torch.Tensor
    from . import ortho6d_torch as _ortho6d_torch
except ImportError:
    pass

_NumpyArray = None
_ortho6d_np = None

try:
    import numpy as np

    _NumpyArray = np.ndarray
    from . import ortho6d_np as _ortho6d_np
except ImportError:
    pass


# Explicit wrapper functions with full documentation
def from_quat(quaternions):
    """
    Convert quaternions to ortho6D representation.

    Parameters
    ----------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]

    Returns
    -------
    ortho6D: torch.Tensor or np.array[..., 3, 2]
        Matrix order: [[r0.x, r0.y],
                       [r1.x, r1.y],
                       [r2.x, r2.y]] where ri is row i.
    """
    if _TorchTensor is not None and isinstance(quaternions, _TorchTensor):
        return _ortho6d_torch.from_quat(quaternions)
    else:
        return _ortho6d_np.from_quat(quaternions)


def from_matrix(rotmats):
    """
    Convert rotation matrices to ortho6D representation.

    Parameters
    ----------
    rotmats : torch.Tensor or np.array[..., 3, 3]
        Matrix order: [[r0.x, r0.y, r0.z],
                       [r1.x, r1.y, r1.z],
                       [r2.x, r2.y, r2.z]] where ri is row i.

    Returns
    -------
    ortho6D: torch.Tensor or np.array[..., 3, 2]
        Matrix order: [[r0.x, r0.y],
                       [r1.x, r1.y],
                       [r2.x, r2.y]] where ri is row i.
    """
    if _TorchTensor is not None and isinstance(rotmats, _TorchTensor):
        return _ortho6d_torch.from_matrix(rotmats)
    else:
        return _ortho6d_np.from_matrix(rotmats)


def to_quat(ortho6D):
    """
    Convert ortho6D to quaternions.

    Parameters
    ----------
    ortho6D: torch.Tensor or np.array[..., 3, 2]
        Matrix order: [[r0.x, r0.y],
                       [r1.x, r1.y],
                       [r2.x, r2.y]] where ri is row i.

    Returns
    -------
    quaternions : torch.Tensor or np.array[..., [w,x,y,z]]
    """
    if _TorchTensor is not None and isinstance(ortho6D, _TorchTensor):
        return _ortho6d_torch.to_quat(ortho6D)
    else:
        return _ortho6d_np.to_quat(ortho6D)


def to_matrix(ortho6D):
    """
    Convert ortho6D to rotation matrices.

    Parameters
    ----------
    ortho6D: torch.Tensor or np.array[..., 3, 2]
        Matrix order: [[r0.x, r0.y],
                       [r1.x, r1.y],
                       [r2.x, r2.y]] where ri is row i.

    Returns
    -------
    rotmats : torch.Tensor or np.array[..., 3, 3]
        Matrix order: [[r0.x, r0.y, r0.z],
                       [r1.x, r1.y, r1.z],
                       [r2.x, r2.y, r2.z]] where ri is row i.
    """
    if _TorchTensor is not None and isinstance(ortho6D, _TorchTensor):
        return _ortho6d_torch.to_matrix(ortho6D)
    else:
        return _ortho6d_np.to_matrix(ortho6D)


# Expose public API
__all__ = [
    "from_quat",
    "from_matrix",
    "to_quat",
    "to_matrix",
]
