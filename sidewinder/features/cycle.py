from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..waveforms import Waveforms


def get_cycles(waveforms: Waveforms, name: str) -> [pd.DataFrame]:
    """Makes a list of the individual cycles from a waveform. This is useful
    for per-cycle feature extraction.

    Each cycle includes the troughs at its start and end.

    Args:
        waveforms: `sidewinder.waveforms.Waveforms` instance holding your data
        name: Name of column in `waveforms` to get cycles for

    Returns:
        Each element contains data from one cycle (e.g. a heartbeat)
    """
    return [
        waveforms.waveforms.iloc[
            waveforms.waveform_features[name]["troughs"][
                cycle_i
            ] : waveforms.waveform_features[name]["troughs"][cycle_i + 1]
            + 1
        ]
        for cycle_i in range(
            waveforms.waveform_features[name]["troughs"].size - 1
        )
    ]


class CycleFeatureExtractor(ABC):
    """Abstract base class for per-cycle (e.g. per heartbeat) feature
    extraction classes."""

    @property
    def class_name(self) -> str:
        """The name of the class itself."""
        return self.__class__.__name__

    @abstractmethod
    def extract_feature(self, waveforms: Waveforms, name: str) -> Waveforms:
        """Extracts a feature for each cycle in a waveform.

        Args:
            waveforms: `sidewinder.waveforms.Waveforms` instance holding your
                data
            name: Name of column in `waveforms` to extract feature from

        Returns:
            `waveforms` with new np.ndarray of shape (n_cycles,) at
                `waveforms.cycle_features[`name`][`self.class_name`]`
                where each element is the feature for the corresponding cycle
        """
        pass


class Duration(CycleFeatureExtractor):
    """Calculates duration (seconds) of each cycle in the waveform."""

    def extract_feature(self, waveforms: Waveforms, name: str) -> Waveforms:
        feature = [
            cycle[waveforms.time_column_name].values[-1]
            - cycle[waveforms.time_column_name].values[0]
            for cycle in get_cycles(waveforms, name)
        ]
        waveforms.cycle_features[name][self.class_name] = np.array(feature)
        return waveforms


class MaximumValue(CycleFeatureExtractor):
    """Calculates maximum value of each cycle in the waveform."""

    def extract_feature(self, waveforms: Waveforms, name: str) -> Waveforms:
        feature = [cycle[name].max() for cycle in get_cycles(waveforms, name)]
        waveforms.cycle_features[name][self.class_name] = np.array(feature)
        return waveforms


class MinimumValue(CycleFeatureExtractor):
    """Calculates minimum value of each cycle in the waveform."""

    def extract_feature(self, waveforms: Waveforms, name: str) -> Waveforms:
        feature = [cycle[name].min() for cycle in get_cycles(waveforms, name)]
        waveforms.cycle_features[name][self.class_name] = np.array(feature)
        return waveforms


class MaximumMinusMinimumValue(CycleFeatureExtractor):
    """Calculates maximum minus minimum value of each cycle in the waveform."""

    def extract_feature(self, waveforms: Waveforms, name: str) -> Waveforms:
        feature = [
            cycle[name].max() - cycle[name].min()
            for cycle in get_cycles(waveforms, name)
        ]
        waveforms.cycle_features[name][self.class_name] = np.array(feature)
        return waveforms


class MeanNegativeFirstDifference(CycleFeatureExtractor):
    """Calculates the mean of only the negative first differences for each cycle
    in the waveform. This feature is equivalent to `mean_dyneg` from
    https://bit.ly/3AwtazE"""

    def extract_feature(self, waveforms: Waveforms, name: str) -> Waveforms:
        feature = [
            np.mean(
                self._remove_zeros(
                    np.clip(
                        np.diff(cycle[name].values, n=1, axis=0),
                        a_min=None,
                        a_max=0,
                    )
                )
            )
            for cycle in get_cycles(waveforms, name)
        ]
        # TODO: See https://github.com/UCL-Chimera/sidewinder/issues/16

        waveforms.cycle_features[name][self.class_name] = np.array(feature)
        return waveforms

    @staticmethod
    def _remove_zeros(array: np.array) -> np.array:
        zeros_removed = array[array < 0]
        if zeros_removed.size > 0:
            return zeros_removed
        # if no negative differences, indicates poor quality waveform
        return np.array([-np.inf])
