from typing import Type

import numpy as np

from ..waveforms import Waveforms
from .cycles import CycleFeatureExtractor


def calculate_diffs(
    wf: Waveforms,
    name: str,
    feature_extractor: Type[CycleFeatureExtractor],
) -> Waveforms:
    fe = feature_extractor()
    if not fe.class_name in wf.features.cycles[name].keys():
        wf = fe.extract_feature(wf, name)
    wf.features.diffs[name][fe.class_name] = np.diff(
        wf.features.cycles[name][fe.class_name]
    )
    return wf
