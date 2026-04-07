# CCS EEG Toolbox (ccstools)

A Python toolbox for EEG analysis at the Center for Cognitive Sciences (CCS), NIMHANS. This toolbox provides functions for PSD estimation, aperiodic component extraction (FOOOF, IRASA), non-linear measures (entropy, fractal dimension), signal processing, and more.

## Features

- **EEG Feature Extraction**: PSD, FOOOF, IRASA, ACW, Catch22, and non-linear complexity measures.
- **Signal Processing**: Filtering, windowing, and synchronization tools.
- **File IO**: Support for various formats including Curry, EDF, and custom formats.
- **Recording**: Tools for real-time EEG recording and manipulation.
- **Plotting**: Visualization for EEG signals and analysis results.

## Installation

To install the toolbox in editable mode (recommended for development):
```bash
git clone https://github.com/arunsasidharan84/ccs_toolbox
cd ccs_toolbox
pip install -e .
```

## Basic Usage

```python
import ccstools
from ccstools.eegfeatures import generate_multieegfeatures

# Example load and process
# data = ... (n_epoch x n_chan x n_samples)
# srate = 500
# chanlist = ['Fz', 'Cz', 'Pz']

# df = generate_multieegfeatures(data, srate, chanlist)
# print(df.head())
```

## Structure

```text
ccstools/
├── ccs_eeg/        # Pipeline and utility functions
├── eegfeatures.py  # Feature extraction (PSD, FOOOF, etc.)
├── sigproc.py      # Signal processing functions
├── fileio.py       # Input/Output help
├── plot.py         # EEG plotting utilities
└── ...             # Other specialized modules
```

## Author

- **Arun Sasidharan** - NIMHANS, Bengaluru.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
