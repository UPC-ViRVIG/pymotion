import numpy as np
import pymotion.rotations.quat as quat
import pymotion.rotations.dual_quat as dquat
import pymotion.ops.vector as vec
from pymotion.ops.forward_kinematics import fk

"""
A skeleton is a set of joints connected by bones.
The skeleton is defined by:
    - the local offsets of the joints
    - the parents of the joints
    - the local rotations of the joints
    - the global position of the root joint
"""


def from_root_positions(positions: np.array, parents: np.array, offsets: np.array) -> np.array:
    """
    Convert the root-centered position space joint positions
    to the skeleton information.
    Note: The joint positions have the global rotation of the root
          applied. Only the root translation should be removed.

    Parameters
    ----------
    positions : np.array[frames, n_joints, 3]
        The root-centered position space (not rotation-relative) joint positions.
    parents : np.array[n_joints]
        The parent of the joint.
    offsets : np.array[n_joints, 3]
        The offset of the joint from its parent.

    Returns
    -------
    rotations : np.array[frames, n_joints, 4]
        The local rotation of the joint.
    """

    nFrames = positions.shape[0]
    nJoints = parents.shape[0]

    # Find all children for each joint:
    children = [[] for _ in range(nJoints)]
    for i, parent in enumerate(parents):
        if i > 0:  # Ensure valid parent index
            children[parent].append(i)

    # Iterate joints and align directions from the rest pose to the predicted pose
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (nFrames, nJoints, 1))
    for j, children_of_j in enumerate(children):
        if len(children_of_j) == 0:  # Skip joints with 0 children
            continue

        # Compute current pose (start with rest pose)
        pos, rotmats = fk(
            rotations,
            np.zeros((1, 3)),
            offsets,
            parents,
        )
        global_rots = quat.from_matrix(rotmats)
        # Align current pose to predicted pose based on the first child
        c = children_of_j[0]
        rest_dir = pos[:, c] - pos[:, j]
        rest_dir = quat.mul_vec(quat.inverse(global_rots[:, j]), rest_dir)
        pred_dir = positions[:, c] - positions[:, j]
        pred_dir = quat.mul_vec(quat.inverse(global_rots[:, j]), pred_dir)
        rot = quat.from_to(rest_dir, pred_dir)
        rotations[:, j] = rot

        # If more than one child, use it for roll correction
        for gc in children_of_j[1:]:
            pos, rotmats = fk(
                rotations,
                np.zeros((1, 3)),
                offsets,
                parents,
            )
            global_rots = quat.from_matrix(rotmats)
            # Align
            rest_gc_dir = pos[:, gc] - pos[:, j]
            rest_gc_dir = quat.mul_vec(quat.inverse(global_rots[:, j]), rest_gc_dir)
            pred_gc_dir = positions[:, gc] - positions[:, j]
            pred_gc_dir = quat.mul_vec(quat.inverse(global_rots[:, j]), pred_gc_dir)
            roll_axis = quat.mul_vec(
                quat.inverse(global_rots[:, j]), vec.normalize(positions[:, c] - positions[:, j])
            )
            roll_rot = quat.from_to_axis(rest_gc_dir, pred_gc_dir, roll_axis)
            rotations[:, j] = quat.mul(rotations[:, j], roll_rot)

    return rotations


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
        The local rotation of the joint.
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
