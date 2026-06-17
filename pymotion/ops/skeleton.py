# Portions Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unified skeleton operations wrapper that dispatches to NumPy or PyTorch backends
based on input tensor types. This provides a single API for skeleton operations
regardless of the underlying array library being used.

A skeleton is a set of joints connected by bones.
The skeleton is defined by:
    - the local offsets of the joints
    - the parents of the joints
    - the local rotations of the joints
    - the global position of the root joint

The dispatcher imports backends at module load time. If a library isn't installed,
its backend will be None and will raise an error at runtime if you try to use it.
"""

from __future__ import annotations

# Import type references and backend modules at module import time
_TorchTensor = None
_skeleton_torch = None

try:
    import torch

    _TorchTensor = torch.Tensor
    from . import skeleton_torch as _skeleton_torch
except ImportError:
    pass

_NumpyArray = None
_skeleton_np = None

try:
    import numpy as np

    _NumpyArray = np.ndarray
    from . import skeleton_np as _skeleton_np
except ImportError:
    pass


# Explicit wrapper functions with full documentation
def fk_quat(rot, global_pos, offsets, parents):
    """
    Compute forward kinematics for a skeleton using quaternion operations.
    From the local rotations, global position and offsets, compute the
    positions and global rotations of the joints in world space.

    This is a memory-efficient alternative to fk() that uses quaternions
    instead of matrices for all computations.

    Parameters
    ----------
    rot : torch.Tensor or np.array[..., n_joints, 4]
        Local rotations as quaternions
    global_pos : torch.Tensor or np.array[..., 3]
        Global position of the root joint
    offsets : torch.Tensor or np.array[..., n_joints, 3] or [n_joints, 3]
        Local offsets of the joints from their parents
    parents : torch.Tensor or np.array[n_joints]
        Parent indices for each joint

    Returns
    -------
    positions : torch.Tensor or np.array[..., n_joints, 3]
        Global positions of the joints in world space
    global_rotations : torch.Tensor or np.array[..., n_joints, 4]
        Global rotations of the joints in world space as quaternions
    """
    if _TorchTensor is not None and isinstance(rot, _TorchTensor):
        return _skeleton_torch.fk_quat(rot, global_pos, offsets, parents)
    else:
        return _skeleton_np.fk_quat(rot, global_pos, offsets, parents)


def fk(rot, global_pos, offsets, parents):
    """
    Compute forward kinematics for a skeleton.
    From the local rotations, global position and offsets, compute the
    positions and rotation matrices of the joints in world space.

    Parameters
    ----------
    rot : torch.Tensor or np.array[..., n_joints, 4]
        Local rotations as quaternions
    global_pos : torch.Tensor or np.array[..., 3]
        Global position of the root joint
    offsets : torch.Tensor or np.array[..., n_joints, 3] or [n_joints, 3]
        Local offsets of the joints from their parents
    parents : torch.Tensor or np.array[n_joints]
        Parent indices for each joint

    Returns
    -------
    positions : torch.Tensor or np.array[..., n_joints, 3]
        Positions of the joints in world space
    rotmats : torch.Tensor or np.array[..., n_joints, 3, 3]
        Rotation matrices of the joints in world space
        Matrix order: [[r0.x, r0.y, r0.z],
                       [r1.x, r1.y, r1.z],
                       [r2.x, r2.y, r2.z]] where ri is row i.
    """
    if _TorchTensor is not None and isinstance(rot, _TorchTensor):
        return _skeleton_torch.fk(rot, global_pos, offsets, parents)
    else:
        return _skeleton_np.fk(rot, global_pos, offsets, parents)


def from_global_rotations(global_quats, parents):
    """
    Compute the inverse forward kinematics for a skeleton.
    From the global rotations and the parents of the joints,
    compute the local rotations of the joints.

    Parameters
    ----------
    global_quats : torch.Tensor or np.array[..., n_joints, 4]
        Global rotations as quaternions
    parents : torch.Tensor or np.array[n_joints]
        Parent indices for each joint

    Returns
    -------
    local_quats : torch.Tensor or np.array[..., n_joints, 4]
        Local rotations of the joints
    """
    if _TorchTensor is not None and isinstance(global_quats, _TorchTensor):
        return _skeleton_torch.from_global_rotations(global_quats, parents)
    else:
        return _skeleton_np.from_global_rotations(global_quats, parents)


def from_root_positions(positions, parents, offsets):
    """
    Convert the root-centered position space joint positions
    to the skeleton information.
    Note: The joint positions have the global rotation of the root
          applied. Only the root translation should be removed.

    Parameters
    ----------
    positions : torch.Tensor or np.array[frames, n_joints, 3]
        The root-centered position space (not rotation-relative) joint positions.
    parents : torch.Tensor or np.array[n_joints]
        The parent of the joint.
    offsets : torch.Tensor or np.array[n_joints, 3]
        The offset of the joint from its parent.

    Returns
    -------
    rotations : torch.Tensor or np.array[frames, n_joints, 4]
        The local rotation of the joint.
    """
    if _TorchTensor is not None and isinstance(positions, _TorchTensor):
        return _skeleton_torch.from_root_positions(positions, parents, offsets)
    else:
        return _skeleton_np.from_root_positions(positions, parents, offsets)


def from_root_dual_quat(dq, parents):
    """
    Convert root-centered dual quaternion to the skeleton information.

    Parameters
    ----------
    dq : torch.Tensor or np.array[..., n_joints, 8]
        Dual quaternions, includes as first element the global position of the root joint
    parents : torch.Tensor or np.array[n_joints]
        Parent indices for each joint

    Returns
    -------
    translations : torch.Tensor or np.array[..., n_joints, 3]
        Local translations of the joints
    rotations : torch.Tensor or np.array[..., n_joints, 4]
        Local rotations of the joints
    """
    if _TorchTensor is not None and isinstance(dq, _TorchTensor):
        return _skeleton_torch.from_root_dual_quat(dq, parents)
    else:
        return _skeleton_np.from_root_dual_quat(dq, parents)


def to_root_dual_quat(rotations, global_pos, parents, offsets):
    """
    Convert the skeleton information to root-centered dual quaternions.

    Parameters
    ----------
    rotations : torch.Tensor or np.array[..., n_joints, 4]
        The local rotation of the joint.
    global_pos : torch.Tensor or np.array[..., 3]
        The global position of the root joint.
    parents : torch.Tensor or np.array[n_joints]
        The parent of the joint.
    offsets : torch.Tensor or np.array[n_joints, 3]
        The offset of the joint from its parent.

    Returns
    -------
    dual_quat : torch.Tensor or np.array[..., n_joints, 8]
        The root-centered dual quaternion representation of the skeleton.
    """
    if _TorchTensor is not None and isinstance(rotations, _TorchTensor):
        return _skeleton_torch.to_root_dual_quat(rotations, global_pos, parents, offsets)
    else:
        return _skeleton_np.to_root_dual_quat(rotations, global_pos, parents, offsets)


def mirror(
    local_rotations,
    global_translation,
    parents,
    offsets,
    end_sites=None,
    joints_mapping=None,
    mode="all",
    axis="X",
):
    """
    Mirror a skeleton along the specified axis. Different modes are available depending on the parameter 'mode'.

    if mode == 'symmetry':
        joints_mapping must be provided, e.g., [0, 1, 3, 2] where 0 and 1 (spine joints) are not swapped,
        3 (right joint) and 2 (left joint) are swapped.
        The topology is not changed, and the joints are mirrored according to the mapping.
        The skeleton must be symmetric w.r.t. the specified axis in the reference pose.
    if mode == 'all':
        This is a perfect mirror, but the topology is also mirrored. joints_mapping is not required.
    if mode == 'positions':
        Positions are mirrored and inverse kinematics is used to compute the local rotations.
        The topology is not mirrored, but the twist of the joints is not preserved. joints_mapping is not required.

    Parameters
    ----------
    local_rotations : torch.Tensor or np.array[..., n_joints, 4]
        The local rotations of the joints.
    global_translation : torch.Tensor or np.array[..., 3]
        The global translation of the root joint.
    parents : torch.Tensor or np.array[n_joints]
        The parent of the joint.
    offsets : torch.Tensor or np.array[n_joints, 3]
        The offset of the joint from its parent.
    end_sites : torch.Tensor or np.array[n_end_sites, 3], optional
        The end sites of the skeleton.
    joints_mapping : torch.Tensor or np.array[n_joints], optional
        The mapping of the joints to mirror. Only required for mode == 'symmetry'.
        joints_mapping must be provided, e.g., [0, 1, 3, 2] where 0 and 1 (spine joints) are not swapped,
        3 (right joint) and 2 (left joint) are swapped.
    mode : str, optional
        The mode of the mirroring: 'symmetry' | 'all' | 'positions'. Default is 'all'.
    axis : str, optional
        The axis to mirror the skeleton along: 'X' | 'Y' | 'Z'. Default is 'X'.

    Returns
    -------
    mirrored_local_rotations : torch.Tensor or np.array[..., n_joints, 4]
        The mirrored local rotations of the joints.
    mirrored_global_translation : torch.Tensor or np.array[..., 3]
        The mirrored global translation of the root joint.
    mirrored_offsets : torch.Tensor or np.array[n_joints, 3]
        The mirrored offset of the joint from its parent.
    mirrored_end_sites : torch.Tensor or np.array[n_end_sites, 3]
        The mirrored end sites of the skeleton.
    """
    if _TorchTensor is not None and isinstance(local_rotations, _TorchTensor):
        return _skeleton_torch.mirror(
            local_rotations,
            global_translation,
            parents,
            offsets,
            end_sites,
            joints_mapping,
            mode,
            axis,
        )
    else:
        return _skeleton_np.mirror(
            local_rotations,
            global_translation,
            parents,
            offsets,
            end_sites,
            joints_mapping,
            mode,
            axis,
        )


# Expose public API
__all__ = [
    "fk",
    "fk_quat",
    "from_global_rotations",
    "from_root_positions",
    "from_root_dual_quat",
    "to_root_dual_quat",
    "mirror",
]
