import torch
import pymotion.rotations.quat_torch as quat


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
