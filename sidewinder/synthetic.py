def calculate_heart_rate(
    target_heart_rate: float,
    hertz: float
) -> (int, float):
    """Adjusts the target heart rate so that one cardiac cycle fits exactly
        within a whole number of timesteps.

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


def calculate_steps_in_phase(
    steps_per_beat: int,
    target_phase_fraction: float
) -> (int, float):
    """Adjusts the target fraction of the cardiac cycle intended for a specific
        phase of that cycle (e.g. time from end-diastole to the systolic peak)
        to occur, so that the phase fits exactly within a whole number of
        timesteps

    Args:
        steps_per_beat: Timesteps per cardiac cycle
        target_phase_fraction: Target fraction of the cardiac cycle for the
            phase to occur in

    Returns:
        Timesteps in the phase
        Adjusted phase fraction
    """
    steps_in_phase = round(steps_per_beat * target_phase_fraction)
    adjusted_phase_fraction = steps_in_phase / steps_per_beat
    return steps_in_phase, adjusted_phase_fraction
