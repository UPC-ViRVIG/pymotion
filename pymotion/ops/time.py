import numpy as np


def interpolate_positions(
    sample_times: np.array,
    original_times: np.array,
    positions: np.array,
    axis: int,
    method: str = "linear",
) -> np.array:
    """
    Perform linear interpolation of positions at specified sample times.

    Parameters
    ----------
    sample_times : np.array
        1D array of times at which to interpolate the positions.
    original_times : np.array
        1D array of times corresponding to the data in `positions`.
    positions : np.array[..., [x, y, z]]
        Positions to interpolate. The array can have any number of dimensions,
        with the positions along the last dimension and the temporal dimension
        specified by `axis`.
    axis : int
        The axis along which the temporal data is stored in `positions`.

    Returns
    -------
    positions : np.array[..., [x, y, z]]
        Interpolated positions. The array has the same shape as `positions`,
        except along the `axis` dimension, where the size is equal to the length
        of `sample_times`.
    """

    assert method == "linear", "Only linear interpolation is supported yet."
    assert (
        positions.shape[axis] == original_times.shape[0]
    ), "Wrong shape of data. Positions along the axis dimension must be equal to the length of original_times."

    # Compute the shapes of the output array
    positions_shape = (
        positions.shape[:axis] + (len(sample_times),) + positions.shape[axis + 1 :]
    )

    # Init array
    out_positions = np.zeros(positions_shape)

    # Compute coefficients for linear interpolation
    idxs = np.minimum(
        np.maximum(np.searchsorted(original_times, sample_times) - 1, 0),
        original_times.shape[0] - 2,
    )
    intervals = original_times[idxs + 1] - original_times[idxs]
    weights = (sample_times - original_times[idxs]) / intervals

    # Use broadcasting to index along the time axis of positions
    selector = [slice(np.newaxis)] * (axis + 1)
    selector.append(Ellipsis)
    selector[axis] = idxs

    # Perform linear interpolation
    out_positions = (1 - weights)[..., np.newaxis] * positions[tuple(selector)]
    selector[axis] = idxs + 1
    out_positions += weights[..., np.newaxis] * positions[tuple(selector)]

    return out_positions
