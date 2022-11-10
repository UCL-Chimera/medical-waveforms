from typing import Type

import numpy as np

from ..waveforms import Waveforms
from .cycles import CycleFeatureExtractor


def calculate_absolute_diffs(
    wf: Waveforms,
    name: str,
    feature_extractor: Type[CycleFeatureExtractor],
) -> Waveforms:
    """Calculates the absolute values of the differences between successive
    values of cycle-level features.

    waveforms.features.diffs[`name`][`feature_extractor.class_name`] is the same
    length as waveforms.features.cycles[`name`][`feature_extractor.class_name`]
    as always has first element 0.0

    Args:
        waveforms: `medical_waveforms.waveforms.Waveforms` instance holding your
            data
        name: Name of column in `waveforms` to extract feature from
        feature_extractor: Extractor class for the cycle-level feature you want
            to calculate the absolute differences for

    Returns:
        `waveforms` with the calculated absolute differences at
            waveforms.features.diffs[`name`][`feature_extractor.class_name`]
    """
    fe = feature_extractor()
    if not fe.class_name in wf.features.cycles[name].keys():
        wf = fe.extract_feature(wf, name)
    wf.features.diffs[name][fe.class_name] = np.zeros(
        wf.features.cycles[name][fe.class_name].size
    )
    wf.features.diffs[name][fe.class_name][1:] = np.abs(
        np.diff(wf.features.cycles[name][fe.class_name])
    )
    return wf
