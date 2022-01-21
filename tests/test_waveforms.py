import pytest
import pandas as pd

from sidewinder import waveforms


class TestWaveforms:
    @pytest.fixture(scope='class')
    def example_data(self):
        return pd.DataFrame({'time': [1, 2, 3], 'signal': [0.1, 0.4, 0.8]})

    def test_validate_waveforms_type(self):
        with pytest.raises(Exception):
            waveforms.Waveforms(waveforms=[1, 2, 3])

    def test_validate_time_column_name(self, example_data):
        with pytest.raises(Exception):
            waveforms.Waveforms(
                waveforms=example_data,
                time_column_name='a_different_time_column_name'
            )

    def test_names(self, example_data):
        w = waveforms.Waveforms(waveforms=example_data)
        assert w.names == ('signal',)
