import numpy as np
from . import quat

"""
Ortho6D are represented as arrays of shape [..., 3, 2]
where the last dimension is the ortho6D representations,
which is the first two columns of the rotation matrix.
Matrix order: [[r0.x, r0.y],
               [r1.x, r1.y],
               [r2.x, r2.y]] where ri is row i.
"""


def from_quat(quaternions: np.array) -> np.array:
    """
    Convert quaternions to otho6D representation.

    Parameters
    ----------
        quat : np.array[..., [w,x,y,z]]

    Returns
    -------
        ortho6D: np.array[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                     [r1.x, r1.y],
                                                     [r2.x, r2.y]] where ri is row i.
    """
    return from_matrix(quat.to_matrix(quaternions))


def from_matrix(rotmats: np.array) -> np.array:
    """
    Convert rotation matrices to ortho6D representation.

    Parameters
    ----------
        rotmats : np.array[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                      [r1.x, r1.y, r1.z],
                                                      [r2.x, r2.y, r2.z]] where ri is row i.

    Returns
    -------
        ortho6D: np.array[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                     [r1.x, r1.y],
                                                     [r2.x, r2.y]] where ri is row i.
    """
    return rotmats[..., :2]


def to_quat(ortho6D: np.array) -> np.array:
    """
    Convert ortho6D to quaternions.

    Parameters
    ----------
        ortho6D: np.array[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                     [r1.x, r1.y],
                                                     [r2.x, r2.y]] where ri is row i.

    Returns
    -------
        quat : np.array[..., [w,x,y,z]]
    """
    return quat.from_matrix(to_matrix(ortho6D))


def to_matrix(ortho6D: np.array) -> np.array:
    """
    Convert ortho6D to rotation matrices.

    Parameters
    ----------
        ortho6D: np.array[..., 3, 2]. Matrix order: [[r0.x, r0.y],
                                                     [r1.x, r1.y],
                                                     [r2.x, r2.y]] where ri is row i.

    Returns
    -------
        rotmats : np.array[..., 3, 3]. Matrix order: [[r0.x, r0.y, r0.z],
                                                      [r1.x, r1.y, r1.z],
                                                      [r2.x, r2.y, r2.z]] where ri is row i.
    """
    c1 = ortho6D[..., 0] / np.linalg.norm(ortho6D[..., 0], axis=-1, keepdims=True)
    c2 = ortho6D[..., 1] - np.sum(c1 * ortho6D[..., 1], axis=-1)[..., np.newaxis] * c1
    c2 = c2 / np.linalg.norm(c2, axis=-1, keepdims=True)
    c3 = np.cross(c1, c2, axis=-1)
    rotations = np.concatenate([c1, c2, c3], axis=-1).reshape(*ortho6D.shape[:-2], 3, 3)
    # transpose last 2 axis
    rotations = np.swapaxes(rotations, -2, -1)
    return rotations
