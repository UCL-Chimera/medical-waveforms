from pyampd import ampd

from ..waveforms import Waveforms


def find_troughs(waveforms: Waveforms, name: str) -> Waveforms:
    """Finds indices of troughs in a waveform.

    Args:
        waveforms: `sidewinder.waveforms.Waveforms` instance holding your data
        name: Name of column in `waveforms` to find troughs in

    Returns:
        `waveforms` with trough indices added to
            `waveforms.features.waveform[`name`]['troughs']`
    """
    waveforms.waveforms[name] *= -1  # invert signal, so troughs become peaks
    waveforms.features.waveform[name]["troughs"] = ampd.find_peaks(
        waveforms.waveforms[name]
    )
    waveforms.waveforms[name] *= -1  # un-invert signal
    return waveforms
