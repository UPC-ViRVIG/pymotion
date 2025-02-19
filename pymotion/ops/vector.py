import numpy as np


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize a vector

    Parameters
    ----------
    v : np.array
    eps : float
        A small epsilon to prevent division by zero.

    Returns
    -------
    normalized_v : np.array
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + eps)
