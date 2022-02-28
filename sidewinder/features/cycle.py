from abc import ABC, abstractmethod

import pandas as pd

from ..waveforms import Waveforms


def get_cycles(waveforms: Waveforms, name: str) -> [pd.DataFrame]:
    """Makes a list of the individual cycles from a waveform. This is useful
        for per-cycle feature extraction

    Args:
        waveforms: `sidewinder.waveforms.Waveforms` instance holding your data
        name: Name of column in `waveforms` to get cycles for

    Returns:
        Each element contains data from one cycle (e.g. a heartbeat)
    """
    return [
        waveforms.waveforms.iloc[
            waveforms.waveform_features[name]['troughs'][cycle_i]:
            waveforms.waveform_features[name]['troughs'][cycle_i + 1]
        ] for cycle_i in range(
            waveforms.waveform_features[name]['troughs'].size - 1
        )
    ]


class CycleFeatureExtractor(ABC):
    """Abstract base class for per-cycle (e.g. per heartbeat) feature
    extraction classes."""
    @abstractmethod
    def extract_feature(self, waveforms: Waveforms, name: str) -> Waveforms:
        """Extracts a feature for each cycle in a waveform.

        Args:
            waveforms: `sidewinder.waveforms.Waveforms` instance holding your
                data
            name: Name of column in `waveforms` to extract feature from

        Returns:
            `waveforms` with trough indices added to
                `waveforms.cycle_features[`name`][`MyFeatureExtractorName`]`
        """
        pass
