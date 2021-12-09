from dataclasses import dataclass

import pandas as pd


@dataclass
class Waveforms:
    """Holds waveforms for downstream processing."""
    waveforms: pd.DataFrame
