import os

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
