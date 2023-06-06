import torch
import pymotion.rotations.quat_torch as quat
import pymotion.rotations.dual_quat_torch as dquat

"""
A skeleton is a set of joints connected by bones.
The skeleton is defined by:
    - the local offsets of the joints
    - the parents of the joints
    - the local rotations of the joints
    - the global position of the root joint

This functions convert skeletal information to 
root-centered dual quaternions and vice-versa.

Root-centered dual quaternions are useful when training
neural networks, as all information is local to the root
and the neural network does not need to learn the FK function.

"""


def from_root_dual_quat(dq: torch.Tensor, parents: torch.Tensor):
    """
    Convert root-centered dual quaternion to the skeleton information.

    Parameters
    ----------
    dq : torch.Tensor[..., n_joints, 8]
        Includes as first element the global position of the root joint
    parents : torch.Tensor[n_joints]

    Returns
    -------
    rotations : torch.Tensor[..., n_joints, 4]
    translations : torch.Tensor[..., n_joints, 3]
    """
    n_joints = dq.shape[-2]
    # rotations has shape (..., frames, n_joints, 4)
    # translations has shape (..., frames, n_joints, 3)
    rotations, translations = dquat.to_rotation_translation(dq.clone())
    # make transformations local to the parents
    # (initially local to the root)
    for j in reversed(range(1, n_joints)):
        parent = parents[j]
        if parent == 0:  # already in root space
            continue
        inv = quat.inverse(rotations[..., parent, :])
        translations[..., j, :] = quat.mul_vec(
            inv,
            translations[..., j, :] - translations[..., parent, :],
        )
        rotations[..., j, :] = quat.mul(inv, rotations[..., j, :])
    return translations, rotations


def to_root_dual_quat(
    rotations: torch.Tensor,
    global_pos: torch.Tensor,
    parents: torch.Tensor,
    offsets: torch.Tensor,
):
    """
    Convert the skeleton information to root-centered dual quaternions.

    Parameters
    ----------
    rotations : torch.Tensor[..., n_joints, 4]
        The rotation of the joint.
    global_pos: torch.Tensor[..., 3]
        The global position of the root joint.
    parents : torch.Tensor[n_joints]
        The parent of the joint.
    offsets : torch.Tensor[n_joints, 3]
        The offset of the joint from its parent.

    Returns
    -------
    dual_quat : torch.Tensor[..., n_joints, 8]
        The root-centered dual quaternion representation of the skeleton.
    """
    assert (offsets[0] == torch.zeros(3)).all()
    n_joints = rotations.shape[1]
    # translations has shape (frames, n_joints, 3)
    rotations = rotations.clone()
    translations = torch.tile(offsets, rotations.shape[:-2] + (1, 1))
    translations[..., 0, :] = global_pos
    # make transformations local to the root
    for j in range(1, n_joints):
        parent = parents[j]
        if parent == 0:  # already in root space
            continue
        translations[..., j, :] = (
            quat.mul_vec(rotations[..., parent, :], translations[..., j, :])
            + translations[..., parent, :]
        )
        rotations[..., j, :] = quat.mul(rotations[..., parent, :], rotations[..., j, :])
    # convert to dual quaternions
    dual_quat = dquat.from_rotation_translation(rotations, translations)
    return dual_quat
