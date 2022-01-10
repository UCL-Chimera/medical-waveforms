def calculate_heart_rate(
    target_heart_rate: float,
    hertz: float
) -> (int, float):
    """Adjusts the heart rate from a target heart rate, so that one cardiac
        cycle fits exactly within a whole number of timesteps.

    Args:
        target_heart_rate: Target heart rate (BPM)
        hertz: Sampling frequency (hertz)

    Returns:
        Timesteps per cardiac cycle
        Adjusted heart rate
    """
    target_seconds_per_beat = 60. / target_heart_rate
    steps_per_beat = round(hertz * target_seconds_per_beat)
    seconds_per_beat = steps_per_beat * (1. / hertz)
    adjusted_heart_rate = 60. / seconds_per_beat
    return steps_per_beat, adjusted_heart_rate
