import os

import numpy as np
import pandas as pd
from numpy.testing import assert_equal
from scipy import interpolate

from medical_waveforms import synthetic, utils


def test_make_waveform_generator_from_file():
    waveform_filepath = os.path.join(
        utils.get_root_directory(),
        "medical_waveforms",
        "data",
        "example_arterial_pressure_waveform.npy",
    )
    generator = synthetic.make_waveform_generator_from_file(waveform_filepath)
    assert isinstance(generator, interpolate.interp1d)


def test_make_generator_timestamps_and_inputs():
    timestamps, inputs = synthetic.make_generator_timestamps_and_inputs(
        cycles_per_minute=60.0, n_cycles_target=2.0, hertz=10.0
    )
    assert_equal(timestamps, np.linspace(0, 2, 21, endpoint=True))
    assert_equal(
        inputs,
        np.concatenate(
            [
                np.linspace(0, 1, 10, endpoint=False),
                np.linspace(0, 1, 11, endpoint=True),
            ]
        ),
    )


def test_synthetic_arterial_pressure_data():
    art = synthetic.synthetic_arterial_pressure_data(
        systolic_pressure=120.0,
        diastolic_pressure=80.0,
        heart_rate=60.0,
        n_beats_target=2.0,
        hertz=10.0,
    )
    assert_equal(art.time.values, np.linspace(0, 2, 21, endpoint=True))
    assert art.pressure.max() == 120.0
    assert art.pressure.min() == 80.0
    assert art.shape == (21, 2)
    assert isinstance(art, pd.DataFrame)
