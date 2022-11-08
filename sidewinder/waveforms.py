from typing import Dict, Tuple

import numpy as np
import pandas as pd


class Waveforms:
    """Holds waveforms for downstream processing."""

    def __init__(
        self, waveforms: pd.DataFrame, time_column_name: str = "time"
    ):
        """
        Args:
            waveforms: Contains a timestamps column (in seconds) and one or more
                waveform columns (arbitrary units)
            time_column_name: The name of the timestamps column in `waveforms`
        """
        self.waveforms = waveforms
        self.time_column_name = time_column_name
        self._validate_arguments()
        self.names = self._init_names()
        self.features = FeaturesContainer(self.names)

    def _validate_arguments(self):
        assert isinstance(
            self.waveforms, pd.DataFrame
        ), "`waveforms` must be a pandas DataFrame"
        assert self.time_column_name in self.waveforms.columns, (
            "`waveforms` must contain a column called "
            f"'{self.time_column_name}'"
        )

    def _init_names(self) -> Tuple[str, ...]:
        """Makes a tuple of names of the waveform-containing columns in
        self.waveforms"""
        return tuple(
            name
            for name in self.waveforms.columns
            if name is not self.time_column_name
        )


class FeaturesContainer:
    """Holds features of the waveform data."""

    def __init__(self, waveform_names: Tuple[str, ...]):
        """
        Args:
            waveform_names: Names of each of the waveform columns
        """
        self.waveform = self._init_features_container(waveform_names)
        self.cycles = self._init_features_container(waveform_names)
        self.diffs = self._init_features_container(waveform_names)

    @staticmethod
    def _init_features_container(
        waveform_names,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Makes a holder for features (the features themselves haven't been
        derived yet)"""
        return {name: {} for name in waveform_names}
