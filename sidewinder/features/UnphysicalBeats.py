# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:22:06 2022

@author: SVG

"""

import numpy as np

from sidewinder.features import cycle
from sidewinder.waveforms import Waveforms


def unphysbeats(waveforms: Waveforms):
    rangeP = [20, 300]  # mmHg
    rangeMAP = [30, 200]  # mmHg
    rangeHR = [20, 200]  # bpm
    rangePP = [20, np.inf]  # mmHg
    # feature_extractor().extract_feature(wf, 'pressure')
    # flag unphysiological beats based on threshold
    for feature_extractor in [
        cycle.CyclesPerMinute,
        cycle.MaximumValue,
        cycle.MinimumValue,
        cycle.MeanValue,
        cycle.MaximumMinusMinimumValue,
        cycle.MeanNegativeFirstDifference,
    ]:
        waveforms = feature_extractor().extract_feature(
            waveforms, "pressure"
        )  ###Picked up from Finn's example.
    Pdias = waveforms.cycle_features["MinimumValue"]
    Psys = waveforms.cycle_features["MaximumValue"]
    MAP = waveforms.cycle_features["MeanValue"]
    PP = waveforms.cycle_features["MaximumMinusMinimumValue"]
    HR = waveforms.cycle_features["CyclesPerMinute"]

    ##Replace without defining the new variables above, if correct
    badP = np.where(np.any(Pdias < rangeP[0] or Psys > rangeP[1]))
    badMAP = np.where(np.any(MAP < rangeMAP[0] or MAP > rangeMAP[1]))
    badHR = np.where(np.any(HR < rangeHR[0] or HR > rangeHR[1]))
    badPP = np.where(np.any(PP < rangePP[0]))

    ###First differences. Smarter way than differencing each time?
    ###Is this a pd dataframe for diff to work?
    fd_Psys = waveforms.cycle_features["Maximumvalue"].diff()
    fd_Pdias = waveforms.cycle_features["Minimumvalue"].diff()
    fd_Period = waveforms.cycle_features["Duration"].diff()
    # fd_Onset=waveforms['Onset'].diff()
    dPsys = 20
    dPdias = 20
    dPeriod = 62.5  # 62.5 samples = 1/2 second
    dPOnset = 20
    noise = -3

    jerkPsys = [
        1 + i for i in range(1, len(fd_Psys)) if abs(fd_Psys[i]) > dPsys
    ]
    jerkPdias = [
        i for i in range(1, len(fd_Pdias)) if abs(fd_Pdias[i]) > dPdias
    ]
    jerkPeriod = [
        1 + i for i in range(1, len(fd_Period)) if abs(fd_Period[i]) > dPeriod
    ]
    # jerkOnset =[1+i for i range (1,len(fd_Onset)) if abs(fd_Onset[i])>dPOnset]

    ##allindices contains all possible bad indiced from above
    allindices = np.concatenate(
        (badP, badMAP, badHR, badPP, jerkPsys, jerkPdias, jerkPeriod)
    )
    bq = [1 if i in allindices else 0 for i in range(0, len(Psys))]
    return bq
