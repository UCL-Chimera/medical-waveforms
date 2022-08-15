# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 17:22:06 2022

@author: SVG

"""

from typing import Dict, Tuple, Type

import numpy as np
from frozendict import frozendict

from sidewinder.features import cycles
from sidewinder.waveforms import Waveforms


def unphysbeats(
    waveforms: Waveforms,
    cycle_thresholds: Dict[
        Type[cycles.CycleFeatureExtractor], Tuple[float, float]
    ] = frozendict(
        {
            cycles.MinimumValue: (20.0, 200.0),  # mmHg
            cycles.MaximumValue: (30.0, 300.0),  # mmHg
            cycles.MeanValue: (30.0, 200.0),  # mmHg
            cycles.CyclesPerMinute: (20.0, 200.0),  # bpm
            cycles.MaximumMinusMinimumValue: (20.0, 250.0),  # mmHg
            cycles.MeanNegativeFirstDifference: (-3.0, 0.0),
        }
    ),
):
    # Flag unphysiological beats based on threshold
    bad_cycles = {}
    for feature_extractor, thresholds in cycle_thresholds.items():
        fe = feature_extractor()
        waveforms = fe.extract_feature(waveforms, "pressure")
        bad_cycles[fe.class_name] = (
            thresholds[0]
            > waveforms.features.cycles["pressure"][fe.class_name]
            > thresholds[1]
        )

    return bad_cycles

    raise NotImplementedError("Function needs revising from here downward")

    ###First differences. Smarter way than differencing each time?
    ###Is this a pd dataframe for diff to work?
    fd_Psys = waveforms.features.cycles["Maximumvalue"].diff()
    fd_Pdias = waveforms.features.cycles["Minimumvalue"].diff()
    fd_Period = waveforms.features.cycles["Duration"].diff()
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
