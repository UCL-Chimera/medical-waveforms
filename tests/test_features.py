import pytest
import numpy as np
import pandas as pd

from sidewinder import synthetic, waveforms
from sidewinder.features import waveform


@pytest.fixture(scope='function')
def abp_data_fixture() -> pd.DataFrame:
    return synthetic.synthetic_arterial_pressure_data(
        systolic_pressure=120.,
        diastolic_pressure=80.,
        heart_rate=60.,
        n_beats_target=2.5,
        hertz=10.
    )


@pytest.fixture(scope='function')
def abp_waveforms_fixture(abp_data_fixture) -> waveforms.Waveforms:
    return waveforms.Waveforms(abp_data_fixture)


def test_find_troughs(abp_waveforms_fixture):
    w = waveform.find_troughs(abp_waveforms_fixture, name='pressure')
    np.testing.assert_array_equal(
        w.waveform_features['pressure']['troughs'],
        np.array([0, 10, 20])
    )
