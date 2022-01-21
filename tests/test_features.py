import numpy as np

from sidewinder import synthetic, waveforms, features


def test_find_troughs():
    data = synthetic.synthetic_arterial_pressure_data(
        systolic_pressure=120.,
        diastolic_pressure=80.,
        heart_rate=60.,
        n_beats_target=2.5,
        hertz=10.
    )
    w = waveforms.Waveforms(data)
    w = features.find_troughs(w, name='pressure')
    np.testing.assert_array_equal(
        w.features['pressure']['troughs'],
        np.array([0, 10, 20])
    )
