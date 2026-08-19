# GW Line Cleaner

Remove line artifacts from gravitational wave data.

A Python package for detecting and removing spectral line artifacts from
gravitational wave detector data. Supports analysis across arbitrary numbers of
detectors (LIGO H1/L1, Virgo, KAGRA, etc.).

## Features

- **Baseline fitting**: Robust polynomial baseline estimation for ASD/PSD
  spectra
- **Line detection**: Automatic detection of spectral lines using peak finding
  algorithms
- **Coherent analysis**: Identify lines appearing across multiple detectors
  (potential astrophysical signals)
- **Line cleaning**: Remove instrument-specific artifacts while preserving
  coherent signals
- **Command-line utility**: `gw-line-cleaner` cleans lines from Short Fourier
  Transform (SFT) files

## Installation

Install from PyPI (when available):

```bash
pip install gw-line-cleaner
```

Or install from source:

```bash
git clone https://github.com/gw-line-cleaner/gw-line-cleaner.git
cd gw-line-cleaner
pip install -e .
```

## Python Interface

### Basic Usage

```python
import gw_line_cleaner as gwlc

# Load and fit baseline for a single detector
frequency, spectrum, result = gwlc.load_and_fit("path/to/spectrum.txt")

# View fitting results
result.info(detector_name="H1")
result.plot(frequency, spectrum, detector_name="H1")

# Detect spectral lines
lines = gwlc.detect_lines(frequency, spectrum, result.baseline, detector_name="H1")
print(f"Detected {len(lines)} lines")
```

### Multi-Detector Coherent Analysis

```python
import gw_line_cleaner as gwlc

# Define detector data files
detector_files = {
    "H1": "path/to/h1_spectrum.txt",
    "L1": "path/to/l1_spectrum.txt",
    "V1": "path/to/v1_spectrum.txt",  # Optional: add more detectors
}

# Analyze coherent lines across detectors
coherent_groups, detector_data = gwlc.analyze_coherent_lines(
    detector_files,
    min_detectors=2  # Lines must appear in at least 2 detectors
)

# Print summary
gwlc.print_summary(coherent_groups, detector_data)

# Plot results
gwlc.plot_results(coherent_groups, detector_data, save_path="coherent_analysis.png")
```

### Line Cleaning

```python
import gw_line_cleaner as gwlc
import numpy as np

# Prepare PSD data for multiple detectors
freq = np.linspace(10, 2000, 100000)
detector_psds = {
    "H1": h1_psd_array,
    "L1": l1_psd_array,
}

# Get cleaned PSDs (incoherent lines removed, coherent lines preserved)
cleaned_psds = gwlc.apply_cleaning(freq, detector_psds, min_detectors=2)

# Or get just the cleaning masks
masks = gwlc.get_cleaning_mask(freq, detector_psds, min_detectors=2)
```

### API Reference

#### Baseline Fitting

- `load(file_path)` - Load spectrum data from file
- `fit(frequency, spectrum)` - Fit baseline to spectrum
- `load_and_fit(file_path)` - Load and fit in one step
- `load_and_fit_multiple(file_dict)` - Process multiple detectors

#### Line Detection

- `detect_lines(frequency, spectrum, baseline)` - Detect spectral lines
- `detect_lines_multiple(detector_data)` - Detect lines for multiple detectors
- `find_and_plot(frequency, spectrum, baseline)` - Detect and visualize

#### Coherent Analysis

- `analyze_coherent_lines(detector_files)` - Full coherent analysis
- `analyze_coherent_lines_from_data(detector_data)` - Analysis from pre-loaded data
- `plot_results(coherent_groups, detector_data)` - Visualize results
- `print_summary(coherent_groups, detector_data)` - Print analysis summary

#### Cleaning

- `clean_lines_from_PSD(freq, detector_psds, detector_clean)` - Core cleaning function
- `get_cleaning_mask(freq, detector_psds)` - Get boolean cleaning masks
- `apply_cleaning(freq, detector_psds)` - Apply cleaning and return cleaned PSDs

## Command Line Utility

The `gw-line-cleaner` utility cleans lines from Short Fourier Transform (SFT,
[SFT]) files, the standard data product utilised by continuous gravitational
data analysis routines from the [LALSuite] library.

### Usage

```
gw-line-cleaner [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-t TEMP_DIR]
                -i INPUT_SFTS -o OUTPUT_SFT_DIR
                [-I INJECT_PARAMS] [-D]
                [-B FSTAT_BEFORE_PARAMS] [-A FSTAT_AFTER_PARAMS]
                [-O OUTPUT_STATS] [-S] [-P OUTPUT_SPECTRA_PLOTS]
                [-e EARTH_EPHEMERIS] [-s SUN_EPHEMERIS]
```

### Options

- **-h, --help**: Show this help message and exit.
- **-l, --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}**: Set the logging
  level (default: INFO).
- **-t, --temp-dir TEMP_DIR**: Temporary directory to write intermediate
  files.
- **-i, --input-SFTs INPUT_SFTS**: Input SFTs. Either a directory containing
  SFT files, or a text file containing SFT filenames.
- **-o, --output-SFT-dir OUTPUT_SFT_DIR**: Directory in which to write output
  SFTs.
- **-I, --inject-params INJECT_PARAMS**: If given, configuration file with
  parameters of a continuous wave signal to be injected into the input SFTs
  before cleaning. If argument contains a colon, the string before the colon is
  the filename, and the string after the colon is the section to read the
  parameters from.
- **-D, --inject-as-depth**: If true, injection amplitudes are interpreted as
  (inverse) sensitivity depths, i.e. *(h0, aPlus, aCross) => (h0, aPlus,
  aCross) * sqrt(Sh)*.
- **-B, --Fstat-before-params FSTAT_BEFORE_PARAMS**: If given, configuration
  file with parameters of a continuous wave signal at which to compute the
  F-statistic before cleaning. If argument contains a colon, the string before
  the colon is the filename, and the string after the colon is the section to
  read the parameters from.
- **-A, --Fstat-after-params FSTAT_AFTER_PARAMS**: If given, configuration
  file with parameters of a continuous wave signal at which to compute the
  F-statistic after cleaning. If argument contains a colon, the string before
  the colon is the filename, and the string after the colon is the section to
  read the parameters from.
- **-O, --output-stats OUTPUT_STATS**: Path to HDF5 file where various
  statistics will be written.
- **-S, --output-spectra**: Include amplitude spectral densities before/after
  cleaning in output statistics.
- **-P, --output-spectra-plots OUTPUT_SPECTRA_PLOTS**: Path to image file
  where plots of amplitude spectral densities before/after cleaning will be
  written.
- **-e, --earth-ephemeris EARTH_EPHEMERIS**: Earth ephemeris file to use.
- **-s, --sun-ephemeris SUN_EPHEMERIS**: Sun ephemeris file to use.

## License

MIT License

[SFT]:          https://dcc.ligo.org/LIGO-T040164/public
[LALSuite]:     https://doi.org/10.7935/GT1W-FZ16
