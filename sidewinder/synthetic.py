import numpy as np
from scipy import interpolate


def make_waveform_generator_from_file(filepath: str) -> interpolate.interp1d:
    """Make scipy interpolation function from saved example waveform file. This
    function can be used to generate resampled synthetic waveforms.

    Args:
        filepath: Path to .npy file containing a 1D numpy array of the example
            waveform

    Returns:
        Interpolation function
    """
    waveform = np.load(filepath)
    time = np.linspace(start=0, stop=1, num=waveform.size)
    return interpolate.interp1d(time, waveform, kind='cubic')
