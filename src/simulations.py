import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("qt5agg")


def make_timestamps(length=10, time_scatter=0, dropout=0):
    """
    Make timestamps for other timeseries generators.
    The timestamps are either uniformly sampled
    with "length" samples (default) or they could
    be moved around and have some dropped out.

    Parameters
    ----------
    length: int
        The number of points in the timeseries.
    time_scatter: float
        If non-zero will move each time measurement around
        using a gaussian distribution with this scatter.
    dropout: float
        Drop out a fraction of the measurements randomly.

    Returns
    -------
    times: np.array, 1D, floats
        The timestamps for the timeseries.
    """
    times = np.linspace(0, length, length, endpoint=False)
    if time_scatter:
        times += np.random.normal(scale=time_scatter, size=times.shape)
    if dropout:
        times = times[np.random.uniform(size=times.shape) > dropout]

    return times


def few_hot_series(length=20, period=4, time_offset=2, time_scatter=0, dropout=0):
    """
    Make a timeseries with zeros everywhere except
    for a few locations, that repeat periodically,
    where the values are 1.

    It is assumed that the "length" parameter traces
    time in arbitrary units, the same as "period".
    I.e., you can imagine that length=10 and period=3.5
    mean we have 10 measurement, one each day, and a
    periodic signal that repeats every 3.5 days.
    Replace "day" with any unit.
    The time_offset and time_scatter are in the same units.

    Use time_scatter and dropout to make this
    a non-uniformly sampled timeseries.

    Parameters
    ----------
    length: int
        The number of points in the timeseries.
    period: float
        The period of repeating "hot measurements".
    time_offset: float
        The position in time of the first measurement.
        If this is larger than "period" it just gets
        modulo the period.
    time_scatter: float
        If non-zero will move each time measurement around
        using a gaussian distribution with this scatter.
        Measurements that are plus/minus one from the
        period will still remain "hot" (have value 1)
        but measurements further from this will be "cold" (0).
        Default is zero (uniformly sampled data).
    dropout: float
        Drop out a fraction of the measurements randomly.


    Returns
    -------
    times: np.array, 1D, float
        The time stamps for this time-series.
    values: np.array, 1D, float
        The values of this time-series.

    """

    times = make_timestamps(length=length, time_scatter=time_scatter, dropout=dropout)

    # make the "hot" measurements
    values = np.zeros_like(times)
    values[np.abs(np.mod(times - time_offset, period)) < 1] = 1

    return times, values


def gaussian_series(
    length=20, period=4, sigma=1, time_offset=2, time_scatter=0, dropout=0
):
    """
    Make a periodic signal with a gaussian profile.

    Parameters
    ----------
    length: int
        The number of points in the timeseries.
    period: float
        The period of repeating peaks.
    sigma: float
        The width of the gaussian profile.
    time_offset: float
        The position in time of the first measurement.
        If this is larger than "period" it just gets
        modulo the period.
    time_scatter: float
        If non-zero will move each time measurement around
        using a gaussian distribution with this scatter.
        Measurements that are plus/minus one from the
        period will still remain "hot" (have value 1)
        but measurements further from this will be "cold" (0).
        Default is zero (uniformly sampled data).
    dropout: float
        Drop out a fraction of the measurements randomly.

    Returns
    -------
    times: np.array, 1D, float
        The time stamps for this time-series.
    values: np.array, 1D, float
        The values of this time-series.
    """
    times = make_timestamps(length=length, time_scatter=time_scatter, dropout=dropout)

    # make the "hot" measurements
    values = np.exp(-0.5 * np.power((np.mod(times, period) - time_offset) / sigma, 2))

    return times, values


def gaussian_template(bins=20, sigma=1, binsize=1, norm=0):
    """
    Make a gaussian template.

    Parameters
    ----------
    bins: int
        The number of points in the template.
    sigma: float
        The width of the gaussian profile.
    binsize: float
        The width of each bin. Usually this
        would be period/bins.
    norm: float
        The normalization of the gaussian profile.
        If zero, the peak is equal to one (default).
        If one, normalize so the sum is equal one.
        If two, normalize so the sqrt of the sum of squares is equal one.

    Returns
    -------
    template: np.array, 1D, float
        The template.
    """
    sigma /= binsize  # adjust the width to units of the bin width
    t = np.linspace(-bins / 2, bins / 2, bins) + 0.5
    template = np.exp(-0.5 * np.power(t / sigma, 2))
    if norm == 1:
        template /= np.sum(template)
    elif norm == 2:
        template /= np.sqrt(np.sum(template**2))
    return template


def sinus_series(length=20, period=4, time_offset=2, time_scatter=0, dropout=0):
    """
    Make a periodic signal with a sinusoidal profile.

    Parameters
    ----------
    length: int
        The number of points in the timeseries.
    period: float
        The period of repeating peaks.
    time_offset: float
        The position in time of the first measurement.
        If this is larger than "period" it just gets
        modulo the period.
    time_scatter: float
        If non-zero will move each time measurement around
        using a gaussian distribution with this scatter.
        Measurements that are plus/minus one from the
        period will still remain "hot" (have value 1)
        but measurements further from this will be "cold" (0).
        Default is zero (uniformly sampled data).
    dropout: float
        Drop out a fraction of the measurements randomly.

    Returns
    -------
    times: np.array, 1D, float
        The time stamps for this time-series.
    values: np.array, 1D, float
        The values of this time-series.
    """
    times = make_timestamps(length=length, time_scatter=time_scatter, dropout=dropout)

    # make the "hot" measurements
    values = np.sin(2 * np.pi * (times - time_offset) / period)

    return times, values


if __name__ == "__main__":
    number = 200
    period = 84

    times, values = few_hot_series(
        length=number, period=period, time_offset=20, time_scatter=0, dropout=0
    )
    plt.bar(times, values, width=1, color="k", align="center")

    times, values = gaussian_series(
        length=number, period=period, sigma=5, time_offset=20, time_scatter=0, dropout=0
    )
    plt.plot(times, values, "-x")

    # times, values = sinus_series(length=number, period=period, time_offset=2, time_scatter=0, dropout=0)
    # plt.plot(times, values, "--")

    plt.show(block=True)

    # from src import folding
    # values += np.random.normal(0, 0.1, len(values))
    # fold = folding.classical_fold(times, values, period, bins=20)
    #
    # print(fold[0,0,:])
    # plt.plot(fold[0,0,:])
    # plt.show(block=True)
