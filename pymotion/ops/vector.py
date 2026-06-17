# Portions Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unified vector operations wrapper that dispatches to NumPy or PyTorch backends
based on input tensor types. This provides a single API for vector operations
regardless of the underlying array library being used.

The dispatcher imports backends at module load time. If a library isn't installed,
its backend will be None and will raise an error at runtime if you try to use it.
"""

from __future__ import annotations

# Import type references and backend modules at module import time
_TorchTensor = None
_vector_torch = None

try:
    import torch

    _TorchTensor = torch.Tensor
    from . import vector_torch as _vector_torch
except ImportError:
    pass

_NumpyArray = None
_vector_np = None

try:
    import numpy as np

    _NumpyArray = np.ndarray
    from . import vector_np as _vector_np
except ImportError:
    pass


def normalize(v, eps=1e-8):
    """
    Normalize a vector

    Parameters
    ----------
    v : torch.Tensor or np.array[..., [x,y,z]]
        Input vector(s) to normalize
    eps : float, optional
        A small epsilon to prevent division by zero. Default is 1e-8.

    Returns
    -------
    normalized_v : torch.Tensor or np.array[..., [x,y,z]]
        Normalized vector(s)
    """
    if _TorchTensor is not None and isinstance(v, _TorchTensor):
        return _vector_torch.normalize(v, eps)
    else:
        return _vector_np.normalize(v, eps)


# Expose public API
__all__ = [
    "normalize",
]
