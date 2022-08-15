from typing import Dict, Tuple, Type

import pandas as pd
from frozendict import frozendict

from sidewinder.features import cycles, diffs
from sidewinder.waveforms import Waveforms


# TODO: These are defaults for adult arterial pressure - move them elsewhere
def check_cycles(
    waveforms: Waveforms,
    name: str,
    cycle_thresholds: Dict[
        Type[cycles.CycleFeatureExtractor], Tuple[float, float]
    ] = frozendict(
        {
            cycles.MinimumValue: (20.0, 200.0),  # mmHg
            cycles.MaximumValue: (30.0, 300.0),  # mmHg
            cycles.MeanValue: (30.0, 200.0),  # mmHg
            cycles.CyclesPerMinute: (20.0, 200.0),  # bpm
            cycles.MaximumMinusMinimumValue: (20.0, 250.0),  # mmHg
            cycles.MeanNegativeFirstDifference: (-3.0, 0.0),
        }
    ),
    diff_thresholds: Dict[
        Type[cycles.CycleFeatureExtractor], float
    ] = frozendict(
        {
            cycles.MinimumValue: 20.0,  # mmHg
            cycles.MaximumValue: 20.0,  # mmHg
            cycles.Duration: 0.5,  # seconds
        }
    ),
) -> pd.DataFrame:
    checked = {}

    # Flag unphysiological beats
    for feature_extractor, thresholds in cycle_thresholds.items():
        fe = feature_extractor()
        waveforms = fe.extract_feature(waveforms, name)
        checked[fe.class_name] = (
            waveforms.features.cycles[name][fe.class_name] > thresholds[0]
        ) & (waveforms.features.cycles[name][fe.class_name] < thresholds[1])

    # Flag unphysiological beat-to-beat changes
    for feature_extractor, threshold in diff_thresholds.items():
        fe = feature_extractor()
        waveforms = diffs.calculate_absolute_diffs(
            waveforms, name, feature_extractor
        )
        checked[f"{fe.class_name}_diff"] = (
            waveforms.features.diffs[name][fe.class_name] < threshold
        )

    checked_df = pd.DataFrame(checked)
    checked_df["all"] = checked_df.all(axis=1)

    return checked_df
