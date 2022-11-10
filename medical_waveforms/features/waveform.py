from typing import Optional
from warnings import warn

from pyampd import ampd

from ..waveforms import Waveforms


def find_troughs(
    waveforms: Waveforms, name: str, scale: Optional[int] = None
) -> Waveforms:
    """Finds indices of troughs in a waveform.

    Args:
        waveforms: `medical_waveforms.waveforms.Waveforms` instance holding your data
        name: Name of column in `waveforms` to find troughs in
        scale : The maximum scale window size is (2 * scale + 1) during trough
            finding. Higher values require more memory. If None, sets `scale`
            to number_of_timesteps // 2 but often much lower values are
            adequate if your data contain many cycle. E.g.
            https://link.springer.com/chapter/10.1007/978-3-319-65798-1_39 uses
            `scale` values around one quarter of the cycle length for#
            intracranial pressure waveforms.

    Returns:
        `waveforms` with trough indices added to
            `waveforms.features.waveform[`name`]['troughs']`
    """
    waveforms.waveforms[name] *= -1  # invert signal, so troughs become peaks

    try:
        waveforms.features.waveform[name]["troughs"] = ampd.find_peaks(
            x=waveforms.waveforms[name], scale=scale
        )
    except MemoryError:
        warn("Ran out of memory. Try setting `scale` to a lower value.")

    waveforms.waveforms[name] *= -1  # un-invert signal
    return waveforms
