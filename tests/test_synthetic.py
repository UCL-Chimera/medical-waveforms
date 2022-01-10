import pytest

from sidewinder import synthetic


@pytest.mark.parametrize(
    "target_heart_rate, hertz, expected",
    [
        (60., 60, (60, 60.)),
        (60.1, 60, (60, 60.)),
    ]
)
def test_calculate_heart_rate(target_heart_rate, hertz, expected):
    assert synthetic.calculate_heart_rate(
        target_heart_rate=target_heart_rate,
        hertz=hertz
    ) == expected


@pytest.mark.parametrize(
    "steps_per_beat, target_phase_fraction, expected",
    [
        (100, 0.1, (10, 0.1)),
        (100, 0.101, (10, 0.1)),
    ]
)
def test_calculate_steps_in_phase(
    steps_per_beat,
    target_phase_fraction,
    expected
):
    assert synthetic.calculate_steps_in_phase(
        steps_per_beat=steps_per_beat,
        target_phase_fraction=target_phase_fraction
    ) == expected
