import torch
import pymotion.rotations.quat_torch as quat
import pymotion.rotations.dual_quat_torch as dquat
import pymotion.ops.vector as vec

"""
A skeleton is a set of joints connected by bones.
The skeleton is defined by:
    - the local offsets of the joints
    - the parents of the joints
    - the local rotations of the joints
    - the global position of the root joint
"""


def fk(
    rot: torch.Tensor,
    global_pos: torch.Tensor,
    offsets: torch.Tensor,
    parents: torch.Tensor,
) -> torch.Tensor:
    """
    Compute forward kinematics for a skeleton.
    From the local rotations, global position and offsets, compute the
    positions and rotation matrices of the joints in world space.

    Parameters
    -----------
        rot: torch.Tensor[..., n_joints, 4]
        global_pos: torch.Tensor[..., 3]
        offsets: torch.Tensor[..., n_joints, 3] or torch.Tensor[n_joints, 3]
        parents: torch.Tensor[n_joints]

    Returns
    --------
        positions: torch.Tensor[..., n_joints, 3]
            positions of the joints
        rotmats: torch.Tensor[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                         [r1.x, r1.y, r1.z],
                                                         [r2.x, r2.y, r2.z]] where ri is row i.
            rotation matrices of the joints
    """
    device = rot.device
    # create a homogeneous matrix of shape (..., 4, 4)
    mat = torch.zeros(
        rot.shape[:-1] + (4, 4),
        device=device,
        dtype=rot.dtype,
    )
    mat[..., :3, :3] = quat.to_matrix(quat.normalize(rot))
    mat[..., :3, 3] = offsets
    mat[..., 3, 3] = 1
    # first joint is global position
    mat[..., 0, :3, 3] = global_pos
    # other joints are transformed by the transform matrix
    for i, parent in enumerate(parents):
        # root
        if i == 0:
            continue
        mat[..., i, :, :] = torch.matmul(
            mat[..., parent, :, :].clone(),
            mat[..., i, :, :].clone(),
        )
    positions = mat[..., :3, 3]
    rotmats = mat[..., :3, :3]
    return positions, rotmats


def from_global_rotations(global_quats: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse forward kinematics for a skeleton.
    From the global rotations and the parents of the joints,
    compute the local rotations of the joints.

    Parameters
    -----------
        global_quats: torch.Tensor[..., n_joints, 4]
        parents: torch.Tensor[n_joints]

    Returns
    --------
        local_quats: torch.Tensor[..., n_joints, 4]
            local rotations of the joints
    """
    local_quats = torch.empty_like(global_quats)

    # Root joint remains the same
    local_quats[..., 0, :] = global_quats[..., 0, :]

    # Precompute inverse of parent rotations
    parent_quats = global_quats[..., parents[1:], :]
    inverse_parent_quats = quat.inverse(parent_quats)

    # Apply inverse parent rotations to child rotations
    child_quats = global_quats[..., 1:, :]
    local_quats[..., 1:, :] = quat.mul(inverse_parent_quats, child_quats)

    return local_quats


def from_root_positions(
    positions: torch.Tensor, parents: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """
    Convert the root-centered position space joint positions
    to the skeleton information.
    Note: The joint positions have the global rotation of the root
          applied. Only the root translation should be removed.

    Parameters
    ----------
    positions : torch.Tensor[frames, n_joints, 3]
        The root-centered position space (not rotation-relative) joint positions.
    parents : torch.Tensor[n_joints]
        The parent of the joint.
    offsets : torch.Tensor[n_joints, 3]
        The offset of the joint from its parent.

    Returns
    -------
    rotations : torch.Tensory[frames, n_joints, 4]
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
    rotations = torch.tile(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=positions.device, dtype=positions.dtype),
        (nFrames, nJoints, 1),
    )
    for j, children_of_j in enumerate(children):
        if len(children_of_j) == 0:  # Skip joints with 0 children
            continue

        # Compute current pose (start with rest pose)
        pos, rotmats = fk(
            rotations,
            torch.zeros((1, 3), device=positions.device, dtype=positions.dtype),
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
                torch.zeros((1, 3), device=positions.device, dtype=positions.dtype),
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
    assert (offsets[0] == torch.zeros(3, device=offsets.device, dtype=offsets.dtype)).all()
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
            quat.mul_vec(rotations[..., parent, :], translations[..., j, :]) + translations[..., parent, :]
        )
        rotations[..., j, :] = quat.mul(rotations[..., parent, :], rotations[..., j, :])
    # convert to dual quaternions
    dual_quat = dquat.from_rotation_translation(rotations, translations)
    return dual_quat


def mirror(
    local_rotations: torch.Tensor,
    global_translation: torch.Tensor,
    parents: torch.Tensor,
    offsets: torch.Tensor,
    end_sites: torch.Tensor = None,
    joints_mapping: torch.Tensor = None,
    mode: str = "all",
    axis: str = "X",
):
    """
    Mirror a skeleton along the X axis. Different modes are available depending on the parameter 'mode'.
    if mode == 'symmetry':
        joints_mapping must be provided, e.g., [0, 1, 3, 2] where 0 and 1 (spine joints) are not swapped, 3 (right joint) and 2 (left joint) are swapped.
        The topology is not changed, and the joints are mirrored according to the mapping.
        The skeleton must be symmetric w.r.t. the X axis in the reference pose.
    if mode == 'all':
        This is a perfect mirror, but the topology is also mirrored. joints_mapping is not required.
    if mode == 'positions':
        Positions are mirrored and inverse kinematics is used to compute the local rotations.
        The topology is not mirrored, but the twist of the joints is not preserved. joints_mapping is not required.

    Parameters
    ----------
    local_rotations : torch.Tensor[..., n_joints, 4]
        The local rotations of the joints.
    global_translation : torch.Tensor[..., 3]
        The global translation of the root joint.
    parents : torch.Tensor[n_joints]
        The parent of the joint.
    offsets : torch.Tensor[n_joints, 3]
        The offset of the joint from its parent.
    end_sites : torch.Tensor[n_end_sites, 3]
        The end sites of the skeleton.
    joints_mapping : torch.Tensor
        The mapping of the joints to mirror. Only required for mode == 'symmetry'.
        joints_mapping must be provided, e.g., [0, 1, 3, 2] where 0 and 1 (spine joints) are not swapped, 3 (right joint) and 2 (left joint) are swapped.
    mode : 'symmetry' | 'all' | 'positions'
        The mode of the mirroring (see above).
    axis : 'X' | 'Y' | 'Z'
        The axis to mirror the skeleton along.

    Returns
    -------
    mirrored_local_rotations : torch.Tensor[..., n_joints, 4]
        The mirrored local rotations of the joints.
    mirrored_global_translation : torch.Tensor[..., 3]
        The mirrored global translation of the root joint.
    mirrored_offsets : torch.Tensor[n_joints, 3]
        The mirrored offset of the joint from its parent.
    mirrored_end_sites : torch.Tensor[n_end_sites, 3]
        The mirrored end sites of the skeleton.

    """

    if mode == "all":
        return _true_mirror(local_rotations, global_translation, parents, offsets, end_sites, axis)

    elif mode == "symmetry":
        if joints_mapping is None:
            raise ValueError("joints_mapping must be provided for mode 'symmetry'")
        if len(joints_mapping) != len(parents):
            raise ValueError("joints_mapping must have the same length as the number of joints")
        if axis == "X":
            mirror_index = 0
            q_mirror = (2, 3)
        elif axis == "Y":
            mirror_index = 1
            q_mirror = (1, 3)
        elif axis == "Z":
            mirror_index = 2
            q_mirror = (1, 2)
        else:
            raise ValueError("Invalid axis. Choose 'X', 'Y', or 'Z'")
        # Convert to global space
        _, global_rotations = fk(local_rotations, torch.zeros_like(global_translation), offsets, parents)
        global_quats = quat.from_matrix(global_rotations)
        # Mirror global positions
        global_translation[..., mirror_index] = -global_translation[..., mirror_index].clone()
        # Mirror X quats
        global_quats = global_quats[..., joints_mapping, :]
        global_quats[..., q_mirror[0]] = -global_quats[..., q_mirror[0]]
        global_quats[..., q_mirror[1]] = -global_quats[..., q_mirror[1]]
        # Convert back to local space
        local_rotations = from_global_rotations(global_quats, parents)
        return local_rotations, global_translation, offsets, end_sites

    elif mode == "positions":
        mirrored_local_rots, mirrored_global_pos, mirrored_offsets, _ = _true_mirror(
            local_rotations, global_translation, parents, offsets, end_sites, axis
        )
        pos, _ = fk(mirrored_local_rots, mirrored_global_pos, mirrored_offsets, parents)
        pos = pos - pos[..., 0:1, :]  # subtract root position to make it root-centered
        mirrored_local_rots = from_root_positions(pos, parents, offsets)
        return mirrored_local_rots, mirrored_global_pos, offsets, end_sites

    else:
        raise ValueError("Invalid mode. Choose 'symmetry', 'all', or 'positions'")


def _true_mirror(
    local_rotations: torch.Tensor,
    global_translation: torch.Tensor,
    parents: torch.Tensor,
    offsets: torch.Tensor,
    end_sites: torch.Tensor = None,
    axis: str = "X",
):
    """
    Mirror a skeleton along the X axis.
    Note: The skeleton topology is also mirrored.

    Parameters
    ----------
    local_rotations : torch.Tensor[..., n_joints, 4]
        The local rotations of the joints.
    global_translation : torch.Tensor[..., 3]
        The global translation of the root joint.
    parents : torch.Tensor[n_joints]
        The parent of the joint.
    offsets : torch.Tensor[n_joints, 3]
        The offset of the joint from its parent.
    end_sites : torch.Tensor[n_end_sites, 3]
        The end sites of the skeleton.
    axis : 'X' | 'Y' | 'Z'
        The axis to mirror the skeleton along.

    Returns
    -------
    mirrored_local_rotations : torch.Tensor[..., n_joints, 4]
        The mirrored local rotations of the joints.
    mirrored_global_translation : torch.Tensor[..., 3]
        The mirrored global translation of the root joint.
    mirrored_offsets : torch.Tensor[n_joints, 3]
        The mirrored offset of the joint from its parent.
    mirrored_end_sites : torch.Tensor[n_end_sites, 3]
        The mirrored end sites of the skeleton.

    """
    local_rotations = local_rotations.clone()
    global_translation = global_translation.clone()
    offsets = offsets.clone()
    if end_sites is not None:
        end_sites = end_sites.clone()

    if axis == "X":
        mirror_index = 0
        q_mirror = (2, 3)
    elif axis == "Y":
        mirror_index = 1
        q_mirror = (1, 3)
    elif axis == "Z":
        mirror_index = 2
        q_mirror = (1, 2)
    else:
        raise ValueError("Invalid axis. Choose 'X', 'Y', or 'Z'")

    # Mirror positions
    offsets[:, mirror_index] = -offsets[:, mirror_index]
    if end_sites is not None:
        end_sites[:, mirror_index] = -end_sites[:, mirror_index]
    global_translation[..., mirror_index] = -global_translation[..., mirror_index]
    # Convert to global space
    _, global_rotations = fk(local_rotations, torch.zeros_like(global_translation), offsets, parents)
    global_quats = quat.from_matrix(global_rotations)
    # Mirror Quats
    global_quats[..., q_mirror[0]] = -global_quats[..., q_mirror[0]]
    global_quats[..., q_mirror[1]] = -global_quats[..., q_mirror[1]]
    # Convert back to local space
    local_rotations = from_global_rotations(global_quats, parents)

    return local_rotations, global_translation, offsets, end_sites
