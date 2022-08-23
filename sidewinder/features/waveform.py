from math import ceil
from statistics import mean

import numpy as np
from pyampd import ampd
from scipy import signal

from ..waveforms import Waveforms


def find_troughs(waveforms: Waveforms, name: str) -> Waveforms:
    """Finds indices of troughs in a waveform.

    Args:
        waveforms: `sidewinder.waveforms.Waveforms` instance holding your data
        name: Name of column in `waveforms` to find troughs in

    Returns:
        `waveforms` with trough indices added to
            `waveforms.waveform_features[`name`]['troughs']`
    """
    waveforms.waveforms[name] *= -1  # invert signal, so troughs become peaks
    waveforms.waveform_features[name]["troughs"] = ampd.find_peaks(
        waveforms.waveforms[name]
    )
    waveforms.waveforms[name] *= -1  # un-invert signal
    return waveforms


def modifed_scholkmann(wave: np.array):
    """
    Implementation of modified scholkmann algorithm
    modified version reduced RAM usage
    https://doi.org/10.1007/978-3-319-65798-1_39

    """

    ##TODO implement optional argument version
    print("starting")
    N = len(wave)
    L = N // 2

    # detrend the data
    # meanval = np.nanmean(wave)
    # wave = np.nan_to_num(wave, copy=False, nan=meanval)
    data = signal.detrend(wave)
    print(data)

    Mx = np.zeros((N, L), dtype=bool)
    Mn = np.zeros((N, L), dtype=bool)

    # print(Mx)

    # local maxima scalogram
    for j in np.arange(1, L):
        for i in range(j + 2, N - j + 1):
            if data[i - 1] > data[i - j - 1] and data[i - 1] > data[i + j - 1]:
                Mx[i - 1, j] = True
            if data[i - 1] < data[i - j - 1] and data[i - 1] < data[i + j - 1]:
                Mn[i - 1, j] = True
    # LSM = np.ones((L, N), dtype=bool)
    # for k in np.arange(1, L + 1):
    #     LSM[k - 1, 0:N - k] &= (data[0:N - k] > data[k:N])  # compare to right neighbours
    #     LSM[k - 1, k:N] &= (data[k:N] > data[0:N - k])  # compare to left neighbours

    print(Mx)
    Y = np.sum(Mx, axis=0)
    print(Y)
    d = Y.argmax(axis=0)
    Mx = Mx.reshape[:, :d]

    Y = np.sum(Mn, axis=0)
    d = Y.argmax(axis=0)
    Mn = Mn.reshape[:, :d]

    Zx = np.sum(Mx, axis=1)
    Zx = np.sum(Mx, axis=1)

    peaks = np.argwhere(Zx=0)
    troughs = np.argwhere(Zn=0)

    maximagram = Mx
    minimagram = Mn

    return (peaks, troughs)


def find_troughs_mod_Scholkmann(waveforms: Waveforms, name: str) -> Waveforms:
    """
    Implementation of modified scholkmann algorithm
    modified version reduced RAM usage
    https://doi.org/10.1007/978-3-319-65798-1_39

    Args:
        waveforms: `sidewinder.waveforms.Waveforms` instance holding your data
        name: Name of column in `waveforms` to find troughs in

    Returns:
        `waveforms` with trough indices added to
            `waveforms.waveform_features[`name`]['troughs']`
    """

    peaks, troughs = modifed_scholkmann(waveforms.waveforms[name])
    waveforms.waveforms[name]["peaks"] = peaks
    waveforms.waveforms[name]["troughs"] = troughs

    return waveforms
