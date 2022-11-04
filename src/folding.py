import numpy as np


def classical_fold(timestamps, values, periods, bins=10):
    """
    Fold the "values" measured at "timestamps" over
    the different "periods" into a number of discrete bins.
    Folding in this context is adding together values that
    were measured at similar times, modulu period.

    Parameters
    ----------
    timestamps: np.array, 1D, floats
        the times when the values were sampled.
    values: np.array, 1D or 2D, floats
        the values (flux, magnitudes, variances, etc.)
        that need to be folded.
        Can be a 2-dimensional array,
        where each value along axis 0 is
        added separately to the corresponding values
        in the same row (each row folded separately,
        on the same timestamps).
        E.g., axis 0 could be flux values
        for different stars taken at the same time
        (in the same image).
        Then axis 1 corresponds to the same
        object/data over time, and must have the
        same length as the timestamps array.
    periods: np.array, 1D, floats, --or-- scalar float
        the periods onto which the folds are calculated,
        e.g., the time on which we calculate the modulu
        for the different timestamps.
    bins: scalar integer
        the number of phase bins that cover the period.
        Default=20, i.e., phase interval of 0.05 of the period.

    Returns
    -------
    folded_values: np.array, 3D, floats
        the folded (summed) values in each phase bin.
        Axis 0 is the same length as axis 0 of "values".
        This is for, e.g., different stars.
        Axis 1 will be for the different periods,
        and will have length equal to the length of "periods".
        The last axis is the phase bin axis, with length equal to "bins".
    """

    if values.ndim == 1:
        # turn 1D array into 2D array
        values = np.expand_dims(values, axis=0)

    if np.isscalar(periods):
        # turn a scalar into a 1D array
        periods = np.array([periods])

    # this is the output of the function:
    folded_values = np.zeros((values.shape[0], len(periods), bins), like=values)

    for (i, p) in enumerate(periods):
        # edges of the phase bins in units of time
        phase_bin_edges = np.linspace(0, p, bins)

        for (j, t) in enumerate(timestamps):
            # the index for the left edge ABOVE/EQUAL to which this sample goes
            idx = np.searchsorted(phase_bin_edges, t % p, side="right") - 1
            folded_values[:, p, idx] += values[:, j]

    return folded_values


if __name__ == "__main__":
    # test the function
    t = np.linspace(0, 10, 100)
    v = np.sin(t)
    p = 1

    f = classical_fold(t, v, p)
    print(f)

    # plot the result
    import matplotlib.pyplot as plt

    plt.plot(t, v, "k.", label="original")
    plt.plot(t, f[0, 0, :], "r.", label="folded")
    plt.legend()
    plt.show()
