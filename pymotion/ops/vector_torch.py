import torch


def normalize(v: torch.tensor, eps: float = 1e-8) -> torch.tensor:
    """
    Normalize a vector

    Parameters
    ----------
    v : torch.tensor
    eps : float
        A small epsilon to prevent division by zero.

    Returns
    -------
    normalized_v : np.array
    """
    norm = torch.linalg.norm(v, dim=-1, keepdims=True)
    return v / (norm + eps)
