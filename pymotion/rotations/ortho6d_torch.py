import torch
from torch.nn import functional as F
from . import quat_torch as quat

"""
Ortho6D are represented as tensors of shape [..., 3, 2]
where the last dimension is the ortho6D representations,
which is the first two columns of the rotation matrix.
Matrix order: [[r0.x, r0.y],
               [r1.x, r1.y],
               [r2.x, r2.y]] where ri is row i.
"""


def from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to otho6D representation.

    Parameters
    ----------
        quat : torch.Tensor[..., [w,x,y,z]]

    Returns
    -------
        ortho6D: torch.Tensor[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                         [r1.x, r1.y],
                                                         [r2.x, r2.y]] where ri is row i.
    """
    return from_matrix(quat.to_matrix(quaternions))


def from_matrix(rotmats: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to ortho6D representation.

    Parameters
    ----------
        rotmats : torch.Tensor[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                          [r1.x, r1.y, r1.z],
                                                          [r2.x, r2.y, r2.z]] where ri is row i.

    Returns
    -------
        ortho6D: torch.Tensor[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                         [r1.x, r1.y],
                                                         [r2.x, r2.y]] where ri is row i.
    """
    return rotmats[..., :2]


def to_quat(ortho6D: torch.Tensor) -> torch.Tensor:
    """
    Convert ortho6D to quaternions.

    Parameters
    ----------
        ortho6D: torch.Tensor[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                         [r1.x, r1.y],
                                                         [r2.x, r2.y]] where ri is row i.
    Returns
    -------
        quat : torch.Tensor[..., [w,x,y,z]]
    """

    return quat.from_matrix(to_matrix(ortho6D))


def to_matrix(ortho6D: torch.Tensor) -> torch.Tensor:
    """
    Convert ortho6D to rotation matrices.

    Parameters
    ----------
        ortho6D: torch.Tensor[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                         [r1.x, r1.y],
                                                         [r2.x, r2.y]] where ri is row i.

    Returns
    -------
        rotmats : torch.Tensor[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                          [r1.x, r1.y, r1.z],
                                                          [r2.x, r2.y, r2.z]] where ri is row i.
    """
    c1 = F.normalize(ortho6D[..., 0], p=2, dim=-1)
    c2 = F.normalize(
        ortho6D[..., 1] - (c1 * ortho6D[..., 1]).sum(-1).unsqueeze(-1) * c1,
        p=2,
        dim=-1,
    )
    c3 = torch.cross(c1, c2, dim=-1)
    rotations = (
        torch.cat([c1, c2, c3], dim=-1)
        .reshape(*ortho6D.shape[:-2], 3, 3)
        .transpose(-2, -1)
    )
    return rotations
