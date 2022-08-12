# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:22:06 2022

@author: SVG

"""

from typing import Dict, Tuple

import numpy as np
from frozendict import frozendict

from sidewinder.features import cycle
from sidewinder.waveforms import Waveforms


def unphysbeats(
    waveforms: Waveforms,
    cycle_thresholds: Dict[str, Tuple[float, float]] = frozendict(
        {
            "Value": (20.0, 300.0),  # mmHg
            "MeanValue": (30.0, 200.0),  # mmHg
            "CyclesPerMinute": (20.0, 200.0),  # bpm
            "MaximumMinusMinimumValues": (20.0, np.inf),  # mmHg
        }
    ),
):
    # TODO: Tweak default thresholds
    # TODO: Separate thresholds for DBP and SBP

    # Flag unphysiological beats based on threshold
    for feature_extractor in [
        cycle.CyclesPerMinute,
        cycle.MaximumValue,
        cycle.MinimumValue,
        cycle.MeanValue,
        cycle.MaximumMinusMinimumValue,
        cycle.MeanNegativeFirstDifference,
    ]:
        waveforms = feature_extractor().extract_feature(waveforms, "pressure")

    # TODO: Do this is a loop
    badP = np.where(
        cycle_thresholds["Value"][0]
        > waveforms.cycle_features["pressure"]["MinimumValue"]
        > cycle_thresholds["Value"][1]
    )
    badMAP = np.where(
        cycle_thresholds["MeanValue"][0]
        > waveforms.cycle_features["pressure"]["MeanValue"]
        > cycle_thresholds["MeanValue"][1]
    )
    badHR = np.where(
        cycle_thresholds["CyclesPerMinute"][0]
        > waveforms.cycle_features["pressure"]["CyclesPerMinute"]
        > cycle_thresholds["CyclesPerMinute"][1]
    )
    badPP = np.where(
        waveforms.cycle_features["pressure"]["MaximumMinusMinimumValue"]
        < cycle_thresholds["MaximumMinusMinimumValues"][0]
    )

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
