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
    positions_shape = positions.shape[:dim] + (len(sample_times),) + positions.shape[dim + 1 :]

    # Init array
    out_positions = torch.zeros(positions_shape, device=device, dtype=positions.dtype)

    # Compute coefficients for linear interpolation
    idxs = torch.min(
        torch.max(
            torch.searchsorted(original_times, sample_times) - 1,
            torch.tensor([0], device=device, dtype=torch.long).expand_as(sample_times),
        ),
        torch.tensor([original_times.shape[0] - 2], device=device, dtype=torch.long).expand_as(sample_times),
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


def savgol_filter(
    x: torch.Tensor,
    window_length: int,
    polyorder: int,
    dim: int = 0,
    mode: str = "nearest",
) -> torch.Tensor:
    """
    Apply a Savitzky-Golay filter to a tensor.

    Parameters
    ----------
    x : torch.Tensor
        The data to be filtered.
    window_length : int
        The length of the filter window. Must be a positive odd integer.
    polyorder : int
        The order of the polynomial used to fit the samples.
        Must be less than `window_length`.
    dim : int, optional
        The dimension along which the filter is applied. Default is 0.
    mode : str, optional
        The padding mode for boundary handling. Currently only "nearest" is
        supported, which replicates edge values. Default is "nearest".

    Returns
    -------
    y : torch.Tensor
        The filtered data, same shape as `x`.
    """
    assert mode == "nearest", "Only 'nearest' mode is supported."
    assert window_length > 0 and window_length % 2 == 1, (
        "window_length must be a positive odd integer."
    )
    assert polyorder < window_length, (
        "polyorder must be less than window_length."
    )

    half_window = window_length // 2
    device = x.device
    dtype = x.dtype

    # Compute Savitzky-Golay coefficients via Vandermonde matrix
    window_pts = torch.arange(
        -half_window, half_window + 1, dtype=torch.float64, device=device
    )
    A = torch.stack([window_pts**k for k in range(polyorder + 1)], dim=1)
    coeffs = torch.linalg.pinv(A)[0]
    coeffs = coeffs.to(dtype)

    # Normalize dim to positive index
    ndim = x.ndim
    dim = dim % ndim

    # Move target dim to the last position
    x_moved = x.movedim(dim, -1)
    original_shape = x_moved.shape
    # Reshape to (batch, length) for conv1d
    x_flat = x_moved.reshape(-1, x_moved.shape[-1])

    # Pad along the last dimension with replicate mode
    # F.pad replicate requires at least 3D input for 1D padding
    x_3d = x_flat.unsqueeze(1)  # (batch, 1, length)
    x_padded = torch.nn.functional.pad(
        x_3d, (half_window, half_window), mode="replicate"
    )

    # Apply 1D convolution
    kernel = coeffs.flip(0).reshape(1, 1, -1)  # (out_channels, in_channels, kW)
    y_3d = torch.nn.functional.conv1d(x_padded, kernel)
    y_flat = y_3d.squeeze(1)

    # Reshape back to original shape and move dim back
    y_moved = y_flat.reshape(original_shape)
    y = y_moved.movedim(-1, dim)
    return y
