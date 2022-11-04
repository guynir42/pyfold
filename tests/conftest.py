import pytest
import numpy as np


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
    times = np.linspace(0, length, length)
    if time_scatter:
        times += np.random.normal(scale=time_scatter, size=times.shape)
    if dropout:
        times = times[np.random.uniform(size=times.shape) > dropout]

    return times


@pytest.fixture
def few_hot_series():
    def _few_hot_series_maker(
        length=10, period=3.5, time_offset=3, time_scatter=0, dropout=0
    ):
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

        times = make_timestamps(
            length=length, time_scatter=time_scatter, dropout=dropout
        )

        # make the "hot" measurements
        values = np.zeros_like(times)
        values[np.abs(np.mod(times - time_offset, period)) < 1] = 1

        return times, values

    return _few_hot_series_maker
