import pytest
import numpy as np
import pandas as pd

from sidewinder import synthetic, waveforms
from sidewinder.features import waveform, cycle


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
    w = waveforms.Waveforms(abp_data_fixture)
    w = waveform.find_troughs(w, name='pressure')
    return w


def test_find_troughs(abp_waveforms_fixture):
    np.testing.assert_array_equal(
        abp_waveforms_fixture.waveform_features['pressure']['troughs'],
        np.array([0, 10, 20])
    )


def test_get_cycles(abp_waveforms_fixture):
    expected = cycle.get_cycles(abp_waveforms_fixture, 'pressure')
    assert len(expected) == 2
    np.testing.assert_array_equal(
        expected[0].pressure.values,
        abp_waveforms_fixture.waveforms.pressure.values[:11]
    )
    np.testing.assert_array_equal(
        expected[1].pressure.values,
        abp_waveforms_fixture.waveforms.pressure.values[10:21]
    )


class TestDuration:
    def test_extract_feature(self, abp_waveforms_fixture):
        duration = cycle.Duration()
        w = duration.extract_feature(abp_waveforms_fixture, 'pressure')
        np.testing.assert_array_equal(
            w.cycle_features['pressure']['Duration'],
            np.array([1., 1.])
        )
