import numpy as np
from . import quat

"""
Dual quaternions are represented as arrays of shape [..., 8]
where the last dimension is the dual quaternion representation.
The first 4 elements are the real part and the last 4 elements are the dual part.
[..., [w_r, x_r, y_r, z_r, w_d, x_d, y_d, z_d]]
"""


def from_rotation_translation(rotations: np.array, translations: np.array) -> np.array:
    """
    Convert the rotations (quaternions) and translation (3D vectors) information to dual quaternions.

    Parameters
    ----------
    rotations: np.array[..., [w, x, y, z]]]
    translations : np.array[..., 3]

    Returns
    -------
    dq : np.array[..., 8]
    """
    # dual quaternion (sigma) = qr + qd (+ is not an addition, but concatenation)
    # real part of dual quaternions represent rotations
    # and is represented as a conventional unit quaternion
    q_r = rotations
    # dual part of dual quaternions represent translations
    # t is a pure quaternion (0, x, y, z)
    # q_d = 0.5 * eps * t * q_r
    t = np.zeros((translations.shape[:-1] + (4,)))
    t[..., 1:] = translations
    q_d = 0.5 * quat.mul(t, q_r)
    dq = np.concatenate((q_r, q_d), axis=-1)
    return dq


def from_translation(translations: np.array) -> np.array:
    """
    Convert a translation to a dual quaternion.

    Parameters
    ----------
    translations : np.array[..., 3]

    Returns
    -------
    dual_quats : np.array[..., 8]
    """
    dual_quats = np.zeros((translations.shape[:-1] + (8,)))
    # real part of dual quaternions represent rotations
    # and is represented as a conventional unit quaternion
    dual_quats[..., 0:1] = 1
    # dual part of dual quaternions represent translations
    # t is a pure quaternion (0, x, y, z)
    # q_d = 0.5 * eps * t * q_r (q_r = 1, thus, q_d = 0.5 * eps * t)
    dual_quats[..., 5:] = translations * 0.5
    return dual_quats


def to_rotation_translation(dq: np.array) -> np.array:
    """
    Convert a dual quaternion to the rotations (quaternions) and translations (3D vectors).

    Parameters
    ----------
    dq: np.array[..., 8]

    Returns
    -------
    rotations: np.array[..., [w, x, y, z]]]
    translations: np.array[..., 3]
    """
    dq = dq.copy()
    q_r = dq[..., :4]
    # rotations can ge get directly from the real part of the dual quaternion
    rotations = q_r
    q_d = dq[..., 4:]
    # the translation (pure quaternion) t = 2 * q_d * q_r*
    # where q_r* is the conjugate of q_r
    translations = (2 * quat.mul(q_d, quat.conjugate(q_r)))[..., 1:]
    return rotations, translations


def normalize(dq: np.array) -> np.array:
    """
    Normalize the dual quaternion to unit length and make sure that
    the dual part is orthogonal to the real part (unit dual quaternion).

    Parameters
    ----------
    dq: np.array[..., 8]

    Returns
    -------
    dq: np.array[..., 8]
    """
    dq = dq.copy()
    q_r = dq[..., :4]
    q_d = dq[..., 4:]
    norm = np.linalg.norm(q_r, axis=-1)
    qnorm = np.stack((norm, norm, norm, norm), axis=-1)
    q_r_normalized = q_r / qnorm
    q_d_normalized = q_d / qnorm
    if not is_unit(np.concatenate((q_r_normalized, q_d_normalized), axis=-1)):
        # make sure that the dual quaternion is orthogonal to the real quaternion
        dot_q_r_q_d = np.sum(q_r * q_d, axis=-1)  # dot product of q_r and q_d
        q_d_normalized_ortho = q_d_normalized - (
            q_r_normalized * (dot_q_r_q_d / (norm * norm))[..., np.newaxis]
        )
        dq = np.concatenate((q_r_normalized, q_d_normalized_ortho), axis=-1)
    else:
        dq = np.concatenate((q_r_normalized, q_d_normalized), axis=-1)
    return dq


def is_unit(dq: np.array, atol: float = 1e-03) -> bool:
    """
    Check if the dual quaternion is a unit one.
    A unit dual quaternion satisfies two properties:
    - The norm of the real part is 1
    - The dot product of the real and dual part is 0.

    Parameters
    ----------
    dq: np.array[..., 8]
    """
    q_r = dq[..., :4]
    q_d = dq[..., 4:]
    sqr_norm_q_r = np.sum(q_r * q_r, axis=-1)
    if np.isclose(sqr_norm_q_r, 0).all():
        return True
    rot_normalized = np.isclose(sqr_norm_q_r, 1).all()
    trans_normalized = np.isclose(np.sum(q_r * q_d, axis=-1), 0, atol=atol).all()
    return rot_normalized and trans_normalized


def unroll(dq: np.array, axis: int) -> np.array:
    """
    Enforce dual quaternion continuity across the time dimension by selecting
    the representation (dq or -dq) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Parameters
    ----------
    dq : np.array[..., 8]
    axis : int
        unroll axis (e.g., frames axis)

    Returns
    -------
    dq : np.array[..., 8]
    """
    dq = dq.swapaxes(0, axis)
    q_r = dq[..., :4]
    # start with the second quaternion since
    # we keep the cover of the first one
    for i in range(1, len(q_r)):
        # distance (dot product) between the previous and current quaternion
        d0 = np.sum(q_r[i] * q_r[i - 1], axis=-1)
        # distance (dot product) between the previous and flipped current quaternion
        d1 = np.sum(-q_r[i] * q_r[i - 1], axis=-1)
        # if the distance with the flipped quaternion is smaller, use it
        dq[i][d0 < d1] = -dq[i][d0 < d1]
    dq = dq.swapaxes(0, axis)
    return dq
