from pyampd import ampd

from ..waveforms import Waveforms


def find_troughs(waveforms: Waveforms, name: str) -> Waveforms:
    """Finds indices of troughs in a waveform.

    Args:
        waveforms: `sidewinder.waveforms.Waveforms` instance holding you data
        name: Name of column in `waveforms` to find troughs in

    Returns:
        `waveforms` with trough indices added to `waveforms.features[`name`]`
    """
    waveforms.waveforms[name] *= -1  # invert signal, so troughs become peaks
    waveforms.features[name]['troughs'] = ampd.find_peaks(
        waveforms.waveforms[name]
    )
    waveforms.waveforms[name] *= -1  # un-invert signal
    return waveforms
