import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use("qt5agg")


def get_periods(timestamps, min_period, bins=20):
    """
    Calculate the periods that are appropriate for the given timestamps,
    based on the minimum period and the number of bins.

    Parameters
    ----------
    timestamps: np.array, 1D, floats
        the times when the values were sampled.
    min_period: scalar float
        the minimum period to be used for the folding.
    bins: scalar integer
        the number of phase bins that cover the period.
        Default=20, i.e., phase interval of 0.05 of the period.

    Returns
    -------
    periods: np.array, 1D, floats
        the periods onto which the folds are calculated,
        e.g., the time on which we calculate the modulu
        for the different timestamps.
    """

    # the time interval between the first and last sample
    dt = max(timestamps) - min(timestamps)

    max_period = min_period * (1 + 1 / bins)
    min_freq = 1 / max_period
    max_freq = 1 / min_period

    phase_width = min_period / bins
    df = phase_width / dt  # df*dt = phase_width

    freqs = np.arange(max_freq, min_freq - df, -df)
    periods = 1 / freqs

    return periods


def classical_fold(timestamps, values, periods, bins=20):
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
        phase_bin_edges = np.linspace(0, p, bins + 1)

        for (j, t) in enumerate(timestamps):
            # the index for the left edge ABOVE/EQUAL to which this sample goes
            idx = np.searchsorted(phase_bin_edges, t % p, side="right") - 1
            folded_values[:, i, idx] += values[:, j]

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
