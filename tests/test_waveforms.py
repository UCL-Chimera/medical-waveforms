import wave

import pandas as pd
import pytest

from sidewinder import waveforms


class TestWaveforms:
    @pytest.fixture(scope="class")
    def example_data(self):
        return pd.DataFrame({"time": [1, 2, 3], "signal": [0.1, 0.4, 0.8]})

    @pytest.fixture(scope="class")
    def example_waveforms(self, example_data):
        return waveforms.Waveforms(waveforms=example_data)

    def test_validate_waveforms_type(self):
        with pytest.raises(Exception):
            waveforms.Waveforms(waveforms=[1, 2, 3])

    def test_validate_time_column_name(self, example_data):
        with pytest.raises(Exception):
            waveforms.Waveforms(
                waveforms=example_data,
                time_column_name="a_different_time_column_name",
            )

    def test_names(self, example_waveforms):
        assert example_waveforms.names == ("signal",)

    def test_features(self, example_waveforms):
        assert example_waveforms.features.waveform == {"signal": {}}
        assert example_waveforms.features.cycles == {"signal": {}}
        assert example_waveforms.features.diffs == {"signal": {}}
