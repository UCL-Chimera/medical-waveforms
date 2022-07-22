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
        self.waveform_features = self._init_features_container()
        self.cycle_features = self._init_features_container()

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

    def _init_features_container(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Makes a holder for features (the features themselves haven't been
        derived yet)"""
        return {name: {} for name in self.names}


class DummyClassForBlackTesting:
    """A dummy class to check that the black GitHub action is working"""




    def __init__(self, argument_with_long_name_1,argument_with_long_name_2,argument_with_long_name_3,argument_with_long_name_4):
        pass  