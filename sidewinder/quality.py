from typing import Optional, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Extra

from sidewinder.features import cycles, diffs
from sidewinder.waveforms import Waveforms


class CycleCheck(BaseModel):
    feature: Type[cycles.CycleFeatureExtractor]
    min: float = -np.inf
    max: float = np.inf
    units: Optional[str] = None
    description: Optional[str] = None


class DiffCheck(BaseModel):
    feature: Type[cycles.CycleFeatureExtractor]
    threshold: float
    units: Optional[str] = None
    description: Optional[str] = None


class ArterialPressureChecks(BaseModel):
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
