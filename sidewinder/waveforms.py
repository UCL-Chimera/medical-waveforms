from typing import Tuple

import pandas as pd


class Waveforms:
    """Holds waveforms for downstream processing."""

    def __init__(self, waveforms: pd.DataFrame, time_column_name: str = 'time'):
        """
        Args:
            waveforms: Contains a timestamps column (in seconds) and one or more
                waveform columns (arbitrary units)
            time_column_name: The name of the timestamps column in `waveforms`
        """
        self.waveforms = waveforms
        self.time_column_name = time_column_name
        self.features = {}
        self._validate_arguments()

    def _validate_arguments(self):
        assert isinstance(self.waveforms, pd.DataFrame), (
            '`waveforms` must be a pandas DataFrame'
        )
        assert self.time_column_name in self.waveforms.columns, (
            "`waveforms` must contain a column called "
            f"'{self.time_column_name}'"
        )

    @property
    def names(self) -> Tuple[str, ...]:
        """The names of the waveform-containing columns in self.waveforms"""
        return tuple(
            name for name in self.waveforms.columns
            if name is not self.time_column_name
        )
