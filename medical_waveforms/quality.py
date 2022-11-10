from typing import Optional, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Extra

from medical_waveforms.features import cycles, diffs
from medical_waveforms.waveforms import Waveforms


class CycleCheck(BaseModel):
    """A signal quality check to run on a feature of each cycle in a signal.

    Args:
        feature: The class that extracts the feature for each cycle
        min: The minimum acceptable value of the feature in order for the check
            to pass for that cycle
        max: The maximum acceptable value of the feature in order for the check
            to pass for that cycle
        units: The units of `min` and `max` (optional and just for
            documentation purposes)
        description: A description of the check (optional and just for
            documentation purposes)
    """

    feature: Type[cycles.CycleFeatureExtractor]
    min: float = -np.inf
    max: float = np.inf
    units: Optional[str] = None
    description: Optional[str] = None


class DiffCheck(BaseModel):
    """A signal quality check to run on the absolute differences between
        features of adjacent cycles in a signal.

    Args:
        feature: The class that extracts the absolute differences feature
        threshold: The minimum acceptable value of absolute difference in order
            for the check to pass for that cycle
        units: The units of `threshold` (optional and just for documentation
            purposes)
        description: A description of the check (optional and just for
            documentation purposes)
    """

    feature: Type[cycles.CycleFeatureExtractor]
    threshold: float
    units: Optional[str] = None
    description: Optional[str] = None


class ArterialPressureChecks(BaseModel):
    """Some preset checks for use with adult human arterial pressure signals."""

    diastolic_pressure: Optional[CycleCheck] = CycleCheck(
        feature=cycles.MinimumValue, min=20.0, max=200.0, units="mmHg"
    )
    systolic_pressure: Optional[CycleCheck] = CycleCheck(
        feature=cycles.MaximumValue, min=30.0, max=300.0, units="mmHg"
    )
    mean_pressure: Optional[CycleCheck] = CycleCheck(
        feature=cycles.MeanValue, min=30.0, max=200.0, units="mmHg"
    )
    heart_rate: Optional[CycleCheck] = CycleCheck(
        feature=cycles.CyclesPerMinute,
        min=20.0,
        max=200.0,
        units="beats per minute",
    )
    pulse_pressure: Optional[CycleCheck] = CycleCheck(
        feature=cycles.MaximumMinusMinimumValue,
        min=20.0,
        max=250.0,
        units="mmHg",
    )
    mean_dyneg: Optional[CycleCheck] = CycleCheck(
        feature=cycles.MeanNegativeFirstDifference,
        min=-3.0,
        max=0.0,
        units="mmHg",
    )
    diastolic_pressure_diff: Optional[DiffCheck] = DiffCheck(
        feature=cycles.MinimumValue, threshold=20.0, units="mmHg"
    )
    systolic_pressure_diff: Optional[DiffCheck] = DiffCheck(
        feature=cycles.MaximumValue, threshold=20.0, units="mmHg"
    )
    beat_time_diff: Optional[DiffCheck] = DiffCheck(
        feature=cycles.Duration, threshold=0.5, units="seconds"
    )

    class Config:
        """Allow users to define extra checks on instantiation."""

        extra = Extra.allow


def check_cycles(
    waveforms: Waveforms, name: str, checks: Type[BaseModel]
) -> pd.DataFrame:
    """Runs signal quality checks for each cycle in a signal.

    Args:
        waveforms: `medical_waveforms.waveforms.Waveforms` instance holding your data
        name: Name of column in `waveforms` to perform signal quality checks on
        checks: The checks to run. This should subclass pydantic's `BaseModel`
            and should have attributes which are instances of `CycleCheck`
            and/or `DiffCheck`, each of which defines a check.

    Returns:
        DataFrame with one row for each cycle in the signal. Has one Boolean
            column for each check, which is True if the check passed for that
            cycle, else False. Also has an 'all' column which is positive if
            all checks passed for that cycle.
    """
    checked = {}

    for check_name, check in vars(checks).items():
        if isinstance(check, CycleCheck):
            # Flag unphysiological cycles
            feature_extractor = check.feature()
            waveforms = check.feature().extract_feature(waveforms, name)
            checked[check_name] = (
                waveforms.features.cycles[name][feature_extractor.class_name]
                > check.min
            ) & (
                waveforms.features.cycles[name][feature_extractor.class_name]
                < check.max
            )

        elif isinstance(check, DiffCheck):
            # Flag unphysiological cycle-to-cycle changes
            feature_extractor = check.feature()
            waveforms = diffs.calculate_absolute_diffs(
                waveforms, name, check.feature
            )
            checked[check_name] = (
                waveforms.features.diffs[name][feature_extractor.class_name]
                < check.threshold
            )

    checked_df = pd.DataFrame(checked)
    checked_df["all"] = checked_df.all(axis=1)

    return checked_df
