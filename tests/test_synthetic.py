import pytest

from sidewinder import synthetic


@pytest.mark.parametrize(
    "target_heart_rate, hertz, expected",
    [
        (60., 60, (60, 60.)),
        (60.1, 60, (60, 60.)),
    ]
)
def test_adjust_target_heart_rate(target_heart_rate, hertz, expected):
    assert synthetic.adjust_target_heart_rate(
        target_heart_rate=target_heart_rate,
        hertz=hertz
    ) == expected
