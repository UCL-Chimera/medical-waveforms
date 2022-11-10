import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal

from medical_waveforms import synthetic, waveforms
from medical_waveforms.features import cycles, diffs, waveform


@pytest.fixture(scope="function")
def abp_data_fixture() -> pd.DataFrame:
    return synthetic.synthetic_arterial_pressure_data(
        systolic_pressure=120.0,
        diastolic_pressure=80.0,
        heart_rate=60.0,
        n_beats_target=2.3,
        hertz=10.0,
    )


@pytest.fixture(scope="function")
def abp_waveforms_fixture(abp_data_fixture) -> waveforms.Waveforms:
    w = waveforms.Waveforms(abp_data_fixture)
    w = waveform.find_troughs(w, name="pressure")
    return w


def test_find_troughs(abp_waveforms_fixture):
    assert_equal(
        abp_waveforms_fixture.features.waveform["pressure"]["troughs"],
        np.array([0, 10, 20]),
    )


def test_get_cycles(abp_waveforms_fixture):
    expected = cycles.get_cycles(abp_waveforms_fixture, "pressure")
    assert len(expected) == 2
    assert_equal(
        expected[0].pressure.values,
        abp_waveforms_fixture.waveforms.pressure.values[:11],
    )
    assert_equal(
        expected[1].pressure.values,
        abp_waveforms_fixture.waveforms.pressure.values[10:21],
    )


class TestDuration:
    def test_extract_feature(self, abp_waveforms_fixture):
        wf = cycles.Duration().extract_feature(
            abp_waveforms_fixture, "pressure"
        )
        assert_equal(
            wf.features.cycles["pressure"]["Duration"], np.array([1.0, 1.0])
        )


class TestCyclesPerMinute:
    def test_extract_feature(self, abp_waveforms_fixture):
        wf = cycles.CyclesPerMinute().extract_feature(
            abp_waveforms_fixture, "pressure"
        )
        assert_equal(
            wf.features.cycles["pressure"]["CyclesPerMinute"],
            np.array([60.0, 60.0]),
        )


class TestMaximumValue:
    def test_extract_feature(self, abp_waveforms_fixture):
        wf = cycles.MaximumValue().extract_feature(
            abp_waveforms_fixture, "pressure"
        )
        assert_equal(
            wf.features.cycles["pressure"]["MaximumValue"],
            np.array([120.0, 120.0]),
        )


class TestMinimumValue:
    def test_extract_feature(self, abp_waveforms_fixture):
        wf = cycles.MinimumValue().extract_feature(
            abp_waveforms_fixture, "pressure"
        )
        assert_equal(
            wf.features.cycles["pressure"]["MinimumValue"],
            np.array([80.0, 80.0]),
        )


class TestMaximumMinusMinimumValue:
    def test_extract_feature(self, abp_waveforms_fixture):
        wf = cycles.MaximumMinusMinimumValue().extract_feature(
            abp_waveforms_fixture, "pressure"
        )
        expected_pulse_pressure = 120.0 - 80.0
        assert_equal(
            wf.features.cycles["pressure"]["MaximumMinusMinimumValue"],
            np.array([expected_pulse_pressure, expected_pulse_pressure]),
        )


class TestMeanValue:
    def test_extract_feature(self, abp_waveforms_fixture):
        wf = cycles.MeanValue().extract_feature(
            abp_waveforms_fixture, "pressure"
        )
        approx_expected_map = 80.0 + (120.0 - 80.0) / 3
        assert_allclose(
            wf.features.cycles["pressure"]["MeanValue"],
            np.array([approx_expected_map, approx_expected_map]),
            atol=1.0,  # allow approximation to be wrong by <1mmHg
        )


class TestMeanNegativeFirstDifference:
    def test_extract_feature(self):
        # Make simple synthetic signal where MNFD is predictable (note that it
        #  contains zeros which will have to be removed). Note that 'time' is
        #  ignored. See See https://github.com/UCL-Chimera/medical-waveforms/issues/16
        data = pd.DataFrame({"time": [0, 1, 2, 3], "signal": [0, 0, 1, 0]})
        wf = waveforms.Waveforms(data)

        # Define manual troughs such that the data contains a single cycle
        wf.features.waveform["signal"]["troughs"] = np.array([0, 3])

        wf = cycles.MeanNegativeFirstDifference().extract_feature(wf, "signal")
        assert_equal(
            wf.features.cycles["signal"]["MeanNegativeFirstDifference"],
            np.array([-1]),
        )


def test_calculate_diffs(abp_waveforms_fixture):
    wf = diffs.calculate_absolute_diffs(
        abp_waveforms_fixture, "pressure", cycles.Duration
    )
    assert_equal(
        wf.features.diffs["pressure"]["Duration"], np.array([0.0, 0.0])
    )
