from lib2to3.pytree import Base

import pandas as pd
import pytest
from pydantic import BaseModel

from sidewinder import quality, synthetic, waveforms
from sidewinder.features import cycles, waveform


class TestArterialPressureChecks:
    def test_instantiation(self):
        """Test if the preset checks are defined validly. If we can instantiate
        without raising an exception, all the pydantic checks must have
        passed."""
        try:
            quality.ArterialPressureChecks()
        except Exception as exc:
            assert (
                False
            ), f"Instantiating ArterialPressureChecks raised an exception {exc}"

    def custom_check(self):
        """Test if user can add a custom check at instantiation."""
        try:
            quality.ArterialPressureChecks(
                map_diff=quality.DiffCheck(
                    feature=cycles.MeanValue, threshold=15.0, units="mmHg"
                )
            )
        except Exception as exc:
            assert False, (
                "Instantiating ArterialPressureChecks with a custom check "
                f"raised an exception {exc}"
            )


@pytest.fixture(scope="function")
def abp_flush_data_fixture() -> pd.DataFrame:
    # Simulate some arterial pressure data
    data_with_flush = synthetic.synthetic_arterial_pressure_data(
        systolic_pressure=120,
        diastolic_pressure=80,
        heart_rate=60,
        n_beats_target=5.3,
        hertz=100,
    )

    # Add a flush-type artifact affecting the 4th and 5th beat (near the
    #  signal's end so that trough finding doesn't fail)
    data_with_flush.loc[330:460, "pressure"] += 300.0

    return data_with_flush


@pytest.fixture(scope="function")
def abp_flush_waveforms_fixture(abp_flush_data_fixture) -> waveforms.Waveforms:
    w = waveforms.Waveforms(abp_flush_data_fixture)
    w = waveform.find_troughs(w, name="pressure")
    return w


def test_check_cycles(abp_flush_waveforms_fixture):
    class SystolicPressureCheck(BaseModel):
        systolic_pressure: quality.CycleCheck = quality.CycleCheck(
            feature=cycles.MaximumValue, min=30.0, max=300.0, units="mmHg"
        )

    check_results = quality.check_cycles(
        abp_flush_waveforms_fixture, "pressure", SystolicPressureCheck()
    )
    expected_columns = ["systolic_pressure", "all"]

    assert list(check_results.columns) == expected_columns
    for column in expected_columns:
        assert (check_results[column].values[:3] == True).all()
        assert (check_results[column].values[3:] == False).all()
