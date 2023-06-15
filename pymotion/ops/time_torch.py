import torch


def interpolate_positions(
    sample_times: torch.Tensor,
    original_times: torch.Tensor,
    positions: torch.Tensor,
    dim: int,
    method: str = "linear",
) -> torch.Tensor:
    """
    Perform linear interpolation of positions at specified sample times.

    Parameters
    ----------
    sample_times : torch.Tensor
        1D tensor of times at which to interpolate the positions.
    original_times : torch.Tensor
        1D tensor of times corresponding to the data in `positions`.
    positions : torch.Tensor[..., [x, y, z]]
        Positions to interpolate. The tensor can have any number of dimensions,
        with the positions along the last dimension and the temporal dimension
        specified by `dim`.
    dim : int
        The dim along which the temporal data is stored in `positions`.

    Returns
    -------
    positions : torch.Tensor[..., [x, y, z]]
        Interpolated positions. The tensor has the same shape as `positions`,
        except along the `dim` dimension, where the size is equal to the length
        of `sample_times`.
    """

    assert method == "linear", "Only linear interpolation is supported yet."
    assert (
        positions.shape[dim] == original_times.shape[0]
    ), "Wrong shape of data. Positions along the dim dimension must be equal to the length of original_times."

    device = positions.device

    # Compute the shapes of the output array
    positions_shape = (
        positions.shape[:dim] + (len(sample_times),) + positions.shape[dim + 1 :]
    )

    # Init array
    out_positions = torch.zeros(positions_shape, device=device)

    # Compute coefficients for linear interpolation
    idxs = torch.min(
        torch.max(
            torch.searchsorted(original_times, sample_times) - 1,
            torch.Tensor([0], device=device).expand_as(sample_times).long(),
        ),
        torch.Tensor([original_times.shape[0] - 2], device=device)
        .expand_as(sample_times)
        .long(),
    )
    intervals = original_times[idxs + 1] - original_times[idxs]
    weights = (sample_times - original_times[idxs]) / intervals

    # Use broadcasting to index along the time axis of positions
    selector = [slice(None)] * (dim + 1)
    selector.append(Ellipsis)
    selector[dim] = idxs

    # Perform linear interpolation
    out_positions = (1 - weights)[..., None] * positions[tuple(selector)]
    selector[dim] = idxs + 1
    out_positions += weights[..., None] * positions[tuple(selector)]

    return out_positions
