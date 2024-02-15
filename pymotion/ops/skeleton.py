import numpy as np
import pymotion.rotations.quat as quat
import pymotion.rotations.dual_quat as dquat

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


def from_root_dual_quat(dq: np.array, parents: np.array) -> np.array:
    """
    Convert root-centered dual quaternion to the skeleton information.

    Parameters
    ----------
    dq: np.array[..., n_joints, 8]
        Includes as first element the global position of the root joint
    parents: np.array[n_joints]

    Returns
    -------
    rotations : np.array[..., n_joints, 4]
    translations : np.array[..., n_joints, 3]
    """
    n_joints = dq.shape[1]
    # rotations has shape (frames, n_joints, 4)
    # translations has shape (frames, n_joints, 3)
    rotations, translations = dquat.to_rotation_translation(dq.copy())
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


def to_root_dual_quat(rotations: np.array, global_pos: np.array, parents: np.array, offsets: np.array):
    """
    Convert the skeleton information to root-centered dual quaternions.

    Parameters
    ----------
    rotations : np.array[..., n_joints, 4]
        The rotation of the joint.
    global_pos: np.array[..., 3]
        The global position of the root joint.
    parents : np.array[n_joints]
        The parent of the joint.
    offsets : np.array[n_joints, 3]
        The offset of the joint from its parent.

    Returns
    -------
    dual_quat : np.array[..., n_joints, 8]
        The root-centered dual quaternion representation of the skeleton.
    """
    assert (offsets[0] == np.zeros(3)).all()
    n_joints = rotations.shape[1]
    # translations has shape (..., n_joints, 3)
    rotations = rotations.copy()
    translations = np.tile(offsets, rotations.shape[:-2] + (1, 1))
    translations[..., 0, :] = global_pos
    # make transformations local to the root
    for j in range(1, n_joints):
        parent = parents[j]
        if parent == 0:  # already in root space
            continue
        translations[..., j, :] = (
            quat.mul_vec(rotations[..., parent, :], translations[..., j, :]) + translations[..., parent, :]
        )
        rotations[..., j, :] = quat.mul(rotations[..., parent, :], rotations[..., j, :])
    # convert to dual quaternions
    dual_quat = dquat.from_rotation_translation(rotations, translations)
    return dual_quat
