import torch
from . import quat_torch as quat

"""
Dual quaternions are represented as tensors of shape [..., 8]
where the last dimension is the dual quaternion representation.
The first 4 elements are the real part and the last 4 elements are the dual part.
[..., [w_r, x_r, y_r, z_r, w_d, x_d, y_d, z_d]]
"""


def from_rotation_translation(
    rotations: torch.Tensor, translations: torch.Tensor
) -> torch.Tensor:
    """
    Convert the rotations (quaternions) and translation (3D vectors) information to dual quaternions.

    Parameters
    ----------
    rotations: torch.Tensor[..., [w, x, y, z]]]
    translations : torch.Tensor[..., 3]

    Returns
    -------
    dq : torch.Tensor[..., 8]
    """
    # dual quaternion (sigma) = qr + qd (+ is not an addition, but concatenation)
    # real part of dual quaternions represent rotations
    # and is represented as a conventional unit quaternion
    q_r = rotations
    # dual part of dual quaternions represent translations
    # t is a pure quaternion (0, x, y, z)
    # q_d = 0.5 * eps * t * q_r
    t = torch.zeros((translations.shape[:-1] + (4,))).to(rotations.device)
    t[..., 1:] = translations
    q_d = 0.5 * quat.mul(t, q_r)
    dq = torch.cat((q_r, q_d), dim=-1)
    return dq


def from_translation(translations: torch.Tensor) -> torch.Tensor:
    """
    Convert a translation to a dual quaternion.

    Parameters
    ----------
    translations : torch.Tensor[..., 3]

    Returns
    -------
    dual_quats : torch.Tensor[..., 8]
    """
    dual_quats = torch.zeros((translations.shape[:-1] + (8,))).to(translations.device)
    # real part of dual quaternions represent rotations
    # and is represented as a conventional unit quaternion
    dual_quats[..., 0:1] = 1
    # dual part of dual quaternions represent translations
    # t is a pure quaternion (0, x, y, z)
    # q_d = 0.5 * eps * t * q_r (q_r = 1, thus, q_d = 0.5 * eps * t)
    dual_quats[..., 5:] = translations * 0.5
    return dual_quats


def to_rotation_translation(dq: torch.Tensor) -> torch.Tensor:
    """
    Convert a dual quaternion to the rotations (quaternions) and translations (3D vectors).

    Parameters
    ----------
    dq: torch.Tensor[..., 8]

    Returns
    -------
    rotations: torch.Tensor[..., [w, x, y, z]]]
    translations: torch.Tensor[..., 3]
    """
    dq = dq.clone()
    q_r = dq[..., :4]
    # rotations can ge get directly from the real part of the dual quaternion
    rotations = q_r
    q_d = dq[..., 4:]
    # the translation (pure quaternion) t = 2 * q_d * q_r*
    # where q_r* is the conjugate of q_r
    translations = (2 * quat.mul(q_d, quat.conjugate(q_r)))[..., 1:]
    return rotations, translations


def normalize(dq: torch.Tensor) -> torch.Tensor:
    """
    Normalize the dual quaternion to unit length and make sure that
    the dual part is orthogonal to the real part (unit dual quaternion).

    Parameters
    ----------
    dq: torch.Tensor[..., 8]

    Returns
    -------
    dq: torch.Tensor[..., 8]
    """
    dq = dq.clone()
    q_r = dq[..., :4]
    q_d = dq[..., 4:]
    norm = torch.linalg.norm(q_r, dim=-1)
    qnorm = torch.stack((norm, norm, norm, norm), dim=-1)
    q_r_normalized = torch.div(q_r, qnorm)
    q_d_normalized = torch.div(q_d, qnorm)
    if not is_unit(torch.cat((q_r_normalized, q_d_normalized), dim=-1)):
        # make sure that the dual quaternion is orthogonal to the real quaternion
        dot_q_r_q_d = torch.sum(q_r * q_d, dim=-1)  # dot product of q_r and q_d
        q_d_normalized_ortho = q_d_normalized - (
            q_r_normalized * torch.div(dot_q_r_q_d, norm * norm).unsqueeze(-1)
        )
        dq = torch.cat((q_r_normalized, q_d_normalized_ortho), dim=-1)
    else:
        dq = torch.cat((q_r_normalized, q_d_normalized), dim=-1)
    return dq


def is_unit(dq: torch.Tensor, atol: float = 1e-03) -> bool:
    """
    Check if the dual quaternion is a unit one.
    A unit dual quaternion satisfies two properties:
    - The norm of the real part is 1
    - The dot product of the real and dual part is 0.

    Parameters
    ----------
    dq: torch.Tensor[..., 8]
    """
    q_r = dq[..., :4]
    q_d = dq[..., 4:]
    sqr_norm_q_r = torch.sum(q_r * q_r, dim=-1)
    if torch.isclose(
        sqr_norm_q_r, torch.zeros(sqr_norm_q_r.shape, device=dq.device)
    ).all():
        return True
    rot_normalized = torch.isclose(
        sqr_norm_q_r, torch.ones(sqr_norm_q_r.shape, device=dq.device)
    ).all()
    sqr_norm_q_d = torch.sum(q_r * q_d, dim=-1)
    trans_normalized = torch.isclose(
        sqr_norm_q_d, torch.zeros(sqr_norm_q_d.shape, device=dq.device), atol=atol
    ).all()
    return rot_normalized and trans_normalized


def unroll(dq: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Enforce dual quaternion continuity across the time dimension by selecting
    the representation (dq or -dq) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Parameters
    ----------
    dq : torch.Tensor[..., 8]
    dim : int
        unroll dimension (e.g., frames dimension)

    Returns
    -------
    dq : torch.Tensor[..., 8]
    """
    dq = dq.swapaxes(0, dim)
    q_r = dq[..., :4]
    # start with the second quaternion since
    # we keep the cover of the first one
    for i in range(1, len(q_r)):
        # distance (dot product) between the previous and current quaternion
        d0 = torch.sum(q_r[i] * q_r[i - 1], dim=-1)
        # distance (dot product) between the previous and flipped current quaternion
        d1 = torch.sum(-q_r[i] * q_r[i - 1], dim=-1)
        # if the distance with the flipped quaternion is smaller, use it
        dq[i][d0 < d1] = -dq[i][d0 < d1]
    dq = dq.swapaxes(0, dim)
    return dq
