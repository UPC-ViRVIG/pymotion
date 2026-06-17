# Portions Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unified center of mass operations wrapper that dispatches to NumPy or PyTorch backends
based on input tensor types. This provides a single API for center of mass operations
regardless of the underlying array library being used.

The dispatcher imports backends at module load time. If a library isn't installed,
its backend will be None and will raise an error at runtime if you try to use it.
"""

from __future__ import annotations

# Import type references and backend modules at module import time
_TorchTensor = None
_center_of_mass_torch = None

try:
    import torch

    _TorchTensor = torch.Tensor
    from . import center_of_mass_torch as _center_of_mass_torch
except ImportError:
    pass

_NumpyArray = None
_center_of_mass_np = None

try:
    import numpy as np

    _NumpyArray = np.ndarray
    from . import center_of_mass_np as _center_of_mass_np
except ImportError:
    pass


def center_of_mass(joints, weights):
    """
    Compute the center of mass of a set of joints.

    Parameters
    ----------
    joints : torch.Tensor or np.array[..., n_joints, 3]
        Joint positions.
    weights : torch.Tensor or np.array[..., n_joints]
        Weights of the joints. The weights should sum to 1 along the last dimension.

    Returns
    -------
    center_of_mass : torch.Tensor or np.array[..., 3]
        Center of mass.
    """
    if _TorchTensor is not None and isinstance(joints, _TorchTensor):
        return _center_of_mass_torch.center_of_mass(joints, weights)
    else:
        return _center_of_mass_np.center_of_mass(joints, weights)


def human_center_of_mass(
    joints_spine,
    joints_left_arm,
    joints_right_arm,
    joints_left_leg,
    joints_right_leg,
):
    """
    Compute the center of mass of a human body defined by the standard human weight distribution.
    Each arm accounts for a 5% of the body weight, each leg accounts for a 15% of the body weight
    and the spine accounts for a 60% of the body weight.

    Parameters
    ----------
    joints_spine : torch.Tensor or np.array[..., n_joints, 3]
        Joint positions of the spine.
    joints_left_arm : torch.Tensor or np.array[..., n_joints, 3]
        Joint positions of the left arm.
    joints_right_arm : torch.Tensor or np.array[..., n_joints, 3]
        Joint positions of the right arm.
    joints_left_leg : torch.Tensor or np.array[..., n_joints, 3]
        Joint positions of the left leg.
    joints_right_leg : torch.Tensor or np.array[..., n_joints, 3]
        Joint positions of the right leg.

    Returns
    -------
    center_of_mass : torch.Tensor or np.array[..., 3]
        Center of mass.
    """
    if _TorchTensor is not None and isinstance(joints_spine, _TorchTensor):
        return _center_of_mass_torch.human_center_of_mass(
            joints_spine,
            joints_left_arm,
            joints_right_arm,
            joints_left_leg,
            joints_right_leg,
        )
    else:
        return _center_of_mass_np.human_center_of_mass(
            joints_spine,
            joints_left_arm,
            joints_right_arm,
            joints_left_leg,
            joints_right_leg,
        )


# Expose public API
__all__ = [
    "center_of_mass",
    "human_center_of_mass",
]
