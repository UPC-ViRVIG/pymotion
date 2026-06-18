# Portions Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unified time operations wrapper that dispatches to NumPy or PyTorch backends
based on input tensor types. This provides a single API for time operations
regardless of the underlying array library being used.

The dispatcher imports backends at module load time. If a library isn't installed,
its backend will be None and will raise an error at runtime if you try to use it.
"""

from __future__ import annotations

# Import type references and backend modules at module import time
_TorchTensor = None
_time_torch = None

try:
    import torch

    _TorchTensor = torch.Tensor
    from . import time_torch as _time_torch
except ImportError:
    pass

_NumpyArray = None
_time_np = None

try:
    import numpy as np

    _NumpyArray = np.ndarray
    from . import time_np as _time_np
except ImportError:
    pass


# Explicit wrapper functions
def interpolate_positions(
    sample_times,
    original_times,
    positions,
    axis=None,
    dim=None,
    method="linear",
):
    """
    Perform linear interpolation of positions at specified sample times.

    Parameters
    ----------
    sample_times : torch.Tensor or np.array
        1D array/tensor of times at which to interpolate the positions.
    original_times : torch.Tensor or np.array
        1D array/tensor of times corresponding to the data in `positions`.
    positions : torch.Tensor or np.array[..., [x, y, z]]
        Positions to interpolate. The array/tensor can have any number of dimensions,
        with the positions along the last dimension and the temporal dimension
        specified by `axis` (for NumPy) or `dim` (for PyTorch).
    axis : int, optional
        The axis along which the temporal data is stored in `positions`.
        Used for NumPy arrays. Either `axis` or `dim` must be provided.
    dim : int, optional
        The dimension along which the temporal data is stored in `positions`.
        Used for PyTorch tensors. Either `axis` or `dim` must be provided.
    method : str, optional
        Interpolation method. Currently only "linear" is supported. Default is "linear".

    Returns
    -------
    positions : torch.Tensor or np.array[..., [x, y, z]]
        Interpolated positions. The array/tensor has the same shape as `positions`,
        except along the temporal dimension, where the size is equal to the length
        of `sample_times`.

    Notes
    -----
    For NumPy arrays, use the `axis` parameter to specify the temporal dimension.
    For PyTorch tensors, use the `dim` parameter to specify the temporal dimension.
    If both are provided, the appropriate one for the tensor type will be used.
    """
    if _TorchTensor is not None and isinstance(sample_times, _TorchTensor):
        # Use dim parameter for PyTorch tensors
        if dim is None:
            if axis is not None:
                dim = axis
            else:
                raise ValueError("Either 'axis' or 'dim' parameter must be provided for PyTorch tensors.")
        return _time_torch.interpolate_positions(sample_times, original_times, positions, dim, method)
    else:
        # Use axis parameter for NumPy arrays
        if axis is None:
            if dim is not None:
                axis = dim
            else:
                raise ValueError("Either 'axis' or 'dim' parameter must be provided for NumPy arrays.")
        return _time_np.interpolate_positions(sample_times, original_times, positions, axis, method)



def savgol_filter(
    x,
    window_length,
    polyorder,
    axis=None,
    dim=None,
    mode="nearest",
):
    """
    Apply a Savitzky-Golay filter to an array or tensor.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        The data to be filtered.
    window_length : int
        The length of the filter window. Must be a positive odd integer.
    polyorder : int
        The order of the polynomial used to fit the samples.
        Must be less than `window_length`.
    axis : int, optional
        The axis along which the filter is applied (NumPy convention).
        Either `axis` or `dim` must be provided.
    dim : int, optional
        The dimension along which the filter is applied (PyTorch convention).
        Either `axis` or `dim` must be provided.
    mode : str, optional
        The padding mode for boundary handling. Currently only "nearest" is
        supported. Default is "nearest".

    Returns
    -------
    y : torch.Tensor or np.ndarray
        The filtered data, same shape as `x`.

    Notes
    -----
    For NumPy arrays, use the `axis` parameter to specify the filter dimension.
    For PyTorch tensors, use the `dim` parameter to specify the filter dimension.
    If both are provided, the appropriate one for the tensor type will be used.
    """
    if _TorchTensor is not None and isinstance(x, _TorchTensor):
        if dim is None:
            if axis is not None:
                dim = axis
            else:
                raise ValueError("Either 'axis' or 'dim' parameter must be provided for PyTorch tensors.")
        return _time_torch.savgol_filter(x, window_length, polyorder, dim, mode)
    else:
        if axis is None:
            if dim is not None:
                axis = dim
            else:
                raise ValueError("Either 'axis' or 'dim' parameter must be provided for NumPy arrays.")
        return _time_np.savgol_filter(x, window_length, polyorder, axis, mode)


# Expose public API
__all__ = [
    "interpolate_positions",
    "savgol_filter",
]
