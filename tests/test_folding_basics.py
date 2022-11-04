import pytest
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("qt5agg")


def test_few_hot_series(few_hot_series):

    number = 100
    period = 3.8
    t, v = few_hot_series(
        length=number, period=period, time_offset=0, time_scatter=0, dropout=0
    )

    assert len(t) == number
    assert len(v) == number
    plt.plot(t, v, "o")
    # plt.show(block=True)
    assert abs(np.sum(v) - int(number / period)) <= 1
    peak_indices = np.where(v)[0]
    peak_times = t[peak_indices]
    dt = np.diff(peak_times)
    print(dt)
