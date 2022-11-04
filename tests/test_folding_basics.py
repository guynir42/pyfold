import pytest
import numpy as np
from src import folding
from src import simulations


def test_folding_ones():
    number = 40
    bins = 10
    t = simulations.make_timestamps(number)
    v = np.ones_like(t)
    period = 10
    fold = folding.classical_fold(t, v, period, bins=bins)
    assert all(fold[0, 0, :] == 4)


def test_few_hot_series():

    number = 100
    period = 3.8
    t, v = simulations.few_hot_series(
        length=number, period=period, time_offset=0, time_scatter=0, dropout=0
    )

    assert len(t) == number
    assert len(v) == number

    assert abs(np.sum(v) - int((number - 1) / period)) <= 1
    peak_indices = np.where(v)[0]
    peak_times = t[peak_indices]
    dt = np.diff(peak_times)
    assert abs(np.mean(dt) - period) <= 1e-2


def test_folding_few_hot():
    number = 20
    period = 4.0
    bins = 10
    t, v = simulations.few_hot_series(
        length=number, period=period, time_offset=2, time_scatter=0, dropout=0
    )
    min_period = 4.0
    p = folding.get_periods(t, min_period, bins=bins)
    print(f"periods= {p}")
    assert np.isclose(min_period, p[0])

    folded = folding.classical_fold(t, v, periods=p, bins=bins)
    assert folded.shape == (1, len(p), bins)

    # each fold should sum to the total flares in the series
    for i in range(len(p)):
        assert np.sum(folded[0, i, :]) == sum(v)

    # the peak of the folded data should contain all
    # the intensity collected from the lightcurve
    assert np.max(folded, axis=None) == sum(v)

    # check that there are no additional periods
    # that can generate a fold different from the
    # ones we already had
    # p2 = np.linspace(min(p), max(p), 200)
    # folded2 = folding.classical_fold(t, v, periods=p2, bins=bins)
    # assert folded2.shape == (1, len(p2), bins)
    # print(folded[0,:,:])
    # print(folded2[0,:,:])
    # for i in range(len(p2)):
    #     matches = np.zeros(len(p))
    #     for j in range(len(p)):
    #         matches[j] = np.all(np.isclose(folded2[0, i, :], folded[0, j, :]))
    #
    #     # at least one row in folded matches this row in folded2
    #     if not any(matches):
    #         print(f'no match for period {p2[i]}')
    #         print(folded2[0, i, :])
    #         print(folded[0, :, :])
    #     assert any(matches)


def test_gaussian_series():
    number = 10000
    sigma = 5
    period = 80
    bins = 20
    t, v = simulations.gaussian_series(
        length=number,
        sigma=sigma,
        period=period,
        time_offset=20,
        time_scatter=0,
        dropout=0,
    )

    assert len(t) == number
    assert len(v) == number

    p = folding.get_periods(t, period, bins=bins)
    fold = folding.classical_fold(t, v, p)

    assert all([abs(np.sum(fold[0, i, :]) - np.sum(v)) <= 1 for i in range(len(p))])

    # find the period closest to the true period:
    idx = np.argmin(np.abs(p - period))

    # find the maximum of each periodogram:
    maxima = np.max(fold[0, :, :], axis=1)

    # the best period has the best maximum:
    assert max(maxima) == np.max(fold[0, idx, :])

    # the S/N for a perfectly matched filter:
    # multiply by the same template as the data
    # sum the total signal, and divide by the
    # sqrt of the sum of squares of the template
    # note the template is equal to the data here
    snr_theoretical = np.sqrt(np.sum(v**2))

    # the S/N for the best period:
    g = simulations.gaussian_template(
        bins=bins, sigma=sigma, binsize=period / bins, norm=2
    )
    assert np.isclose(np.sum(g**2), 1.0)

    # the noise is also stacked many times when folding
    noise_fold = np.sqrt(
        folding.classical_fold(t, np.ones_like(v), p[idx], bins=bins)[0, 0, :]
    )

    # norm_fold = fold[0, idx, :] / noise_fold
    # norm_fold[np.isnan(norm_fold)] = 0

    filtered = np.convolve(g, fold[0, idx, :], mode="same")
    filtered /= np.sqrt(np.sum((g * noise_fold) ** 2))
    snr_measured = np.max(filtered)

    # import matplotlib.pyplot as plt
    # for i in range(len(p)):
    #     plt.plot(fold[0, i, :], '-o', label=f'P= {p[i]}')
    # g_scaled = g * np.max(fold[0, idx, :]) / np.max(g)
    # plt.plot(range(bins), g_scaled, '--x', label='template')
    # plt.legend()
    # plt.show(block=True)

    print(
        f"snr theoretical: {snr_theoretical} | snr measured: {snr_measured} | ratio: {snr_measured / snr_theoretical}"
    )
    assert abs(snr_measured - snr_theoretical) < 2.0
