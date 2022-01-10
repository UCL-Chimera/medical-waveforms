import os

import numpy as np
import pandas as pd
from scipy import interpolate

from sidewinder import utils
from sidewinder import synthetic


def test_make_waveform_generator_from_file():
    waveform_filepath = os.path.join(
        utils.get_root_directory(),
        'sidewinder',
        'data',
        'example_arterial_pressure_waveform.npy'
    )
    generator = synthetic.make_waveform_generator_from_file(waveform_filepath)
    assert isinstance(generator, interpolate.interp1d)


def test_make_generator_timestamps_and_inputs():
    timestamps, inputs = synthetic.make_generator_timestamps_and_inputs(
        cycles_per_minute=60.,
        n_cycles_target=2.,
        hertz=10.
    )
    assert (timestamps == np.linspace(0, 2, 20, endpoint=False)).all()
    assert (inputs == np.tile(np.linspace(0, 1, 10, endpoint=False), 2)).all()


def test_synthetic_arterial_pressure_data():
    art = synthetic.synthetic_arterial_pressure_data(
        systolic_pressure=120.,
        diastolic_pressure=80.,
        heart_rate=60.,
        n_beats_target=2.,
        hertz=10.
    )
    assert (art.time.values == np.linspace(0, 2, 20, endpoint=False)).all()
    assert art.pressure.max() == 120.
    assert art.pressure.min() == 80.
    assert art.shape == (20, 2)
    assert isinstance(art, pd.DataFrame)
