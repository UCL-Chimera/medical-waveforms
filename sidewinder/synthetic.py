import os
import math

import numpy as np
from scipy import interpolate

from .utils import get_root_directory


def make_waveform_generator_from_file(filepath: str) -> interpolate.interp1d:
    """Make scipy interpolation function from saved example waveform file. This
    function can be used to generate resampled synthetic waveforms.

    Args:
        filepath: Path to .npy file containing a 1D numpy array of the example
            waveform

    Returns:
        Interpolation function with domain [0, 1]
    """
    waveform = np.load(filepath)
    time = np.linspace(start=0, stop=1, num=waveform.size)
    return interpolate.interp1d(time, waveform, kind='cubic')


def make_generator_timestamps_and_inputs(
    cycles_per_minute: float,
    n_cycles_target: float,
    hertz: float
) -> (np.ndarray, np.ndarray):
    """Make timestamps and corresponding inputs for a synthetic waveform
        generator function.

    Args:
        cycles_per_minute: Rate (cycles per minute)
        n_cycles_target: Target number of cycles. Doesn't have to be an
            integer. The number of cycles actually generated will be slightly
            different from the target unless each cycle fits exactly within a
            whole number of samples.
        hertz: Sampling rate (hertz)

    Returns:
        Timestamps (seconds)
        Inputs. The units are such that 0 is the start of the generated
            waveform and 1 is the end.
    """
    seconds_per_cycle = 60. / cycles_per_minute
    dt = 1 / hertz
    fractional_samples_per_cycle = seconds_per_cycle / dt
    n_samples = round(fractional_samples_per_cycle * n_cycles_target)
    n_seconds = n_samples * dt
    n_cycles_actual = n_samples / fractional_samples_per_cycle
    n_cycle_starts = math.ceil(n_cycles_actual)

    start_times = np.zeros(n_cycle_starts)
    start_times[1:] = np.cumsum(
        np.repeat(seconds_per_cycle, n_cycle_starts - 1)
    )

    timestamps = np.linspace(0, n_seconds, n_samples, endpoint=False)

    """`insertion_i` is the indices where `start_times` would be inserted into
    `timestamps` to maintain order. Each element in `offsets` is the number
    of seconds that elapse in a cycle before the first sample in that cycle."""
    insertion_i = np.searchsorted(timestamps, start_times)
    offsets = timestamps[insertion_i] - start_times

    samples_per_cycle = np.zeros(n_cycle_starts)
    samples_per_cycle[:-1] = insertion_i[1:] - insertion_i[:-1]
    samples_per_cycle[-1] = n_samples - samples_per_cycle.sum()
    samples_per_cycle = samples_per_cycle.astype(int)

    def cycle_timestamps(cycle_i: int) -> np.ndarray:
        return np.linspace(
            start=0,
            stop=(samples_per_cycle[cycle_i] - 1) / hertz,
            num=samples_per_cycle[cycle_i]
        ) + offsets[cycle_i]

    inputs = np.concatenate([
        cycle_timestamps(cycle_i) for cycle_i in range(n_cycle_starts)
    ]) / seconds_per_cycle

    return timestamps, inputs


def synthetic_arterial_pressure_data(
    systolic_pressure: float,
    diastolic_pressure: float,
    heart_rate: float,
    n_beats_target: float,
    hertz: float
) -> (np.ndarray, np.ndarray):
    assert systolic_pressure > diastolic_pressure

    waveform_filepath = os.path.join(
        get_root_directory(),
        'sidewinder',
        'data',
        'example_arterial_pressure_waveform.npy'
    )
    generator = make_waveform_generator_from_file(waveform_filepath)

    timestamps, inputs = make_generator_timestamps_and_inputs(
        cycles_per_minute=heart_rate,
        n_cycles_target=n_beats_target,
        hertz=hertz
    )
    waveform = generator(inputs)

    # Scale
    waveform -= waveform.min()
    waveform /= waveform.max()
    waveform *= (systolic_pressure - diastolic_pressure)
    waveform += diastolic_pressure

    return timestamps, waveform
