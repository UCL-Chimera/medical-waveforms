![Run tests workflow](https://github.com/UCL-Chimera/medical-waveforms/actions/workflows/run_tests.yml/badge.svg) ![Linting workflow](https://github.com/UCL-Chimera/medical-waveforms/actions/workflows/lint.yml/badge.svg)

# medical-waveforms

**medical-waveforms** is a Python package for preprocessing and analysis of physiological waveforms.

This package currently focuses on:

- Splitting waveforms into individual cycles (e.g. splitting a respiratory waveform into individual breaths)
- Extracting features from individual cycles
- Rule-based signal quality assessment


## Installation

Install with:

```
pip install medical-waveforms
```

The package is tested on Python 3.7 and above.


## Getting started

See the [tutorial notebook](https://github.com/UCL-Chimera/medical-waveforms/blob/main/examples/tutorial.ipynb) for a general introduction to using the package.

The [signal quality assessment notebook](https://github.com/UCL-Chimera/medical-waveforms/blob/main/examples/signal_quality.ipynb) demonstrates customisation of the signal quality assessment process.

These tutorials currently focus on arterial blood pressure waveforms, but can be adapted to other physiological waveforms.


## Contributing to this project

Contributions are very welcome! Please see [CONTRIBUTING.md](https://github.com/UCL-Chimera/medical-waveforms/blob/main/CONTRIBUTING.md) to get started.


## Acknowledgements

Our signal quality assessment pipeline is adapted from that used in the excellent [PhysioNet Cardiovascular Signal Toolbox](https://github.com/cliffordlab/PhysioNet-Cardiovascular-Signal-Toolbox). Many thanks to its [contributors](https://github.com/cliffordlab/PhysioNet-Cardiovascular-Signal-Toolbox/graphs/contributors).
