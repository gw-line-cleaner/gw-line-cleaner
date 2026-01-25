# GW Line Cleaner

Remove line artifacts from gravitational wave data.

A Python package for detecting and removing spectral line artifacts from gravitational wave detector data. Supports analysis across arbitrary numbers of detectors (LIGO H1/L1, Virgo, KAGRA, etc.).

## Features

- **Baseline fitting**: Robust polynomial baseline estimation for ASD/PSD spectra
- **Line detection**: Automatic detection of spectral lines using peak finding algorithms
- **Coherent analysis**: Identify lines appearing across multiple detectors (potential astrophysical signals)
- **Line cleaning**: Remove instrument-specific artifacts while preserving coherent signals

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

## Quick Start

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

## API Reference

### Baseline Fitting

- `load(file_path)` - Load spectrum data from file
- `fit(frequency, spectrum)` - Fit baseline to spectrum
- `load_and_fit(file_path)` - Load and fit in one step
- `load_and_fit_multiple(file_dict)` - Process multiple detectors

### Line Detection

- `detect_lines(frequency, spectrum, baseline)` - Detect spectral lines
- `detect_lines_multiple(detector_data)` - Detect lines for multiple detectors
- `find_and_plot(frequency, spectrum, baseline)` - Detect and visualize

### Coherent Analysis

- `analyze_coherent_lines(detector_files)` - Full coherent analysis
- `analyze_coherent_lines_from_data(detector_data)` - Analysis from pre-loaded data
- `plot_results(coherent_groups, detector_data)` - Visualize results
- `print_summary(coherent_groups, detector_data)` - Print analysis summary

### Cleaning

- `clean_lines_from_PSD(freq, detector_psds, detector_clean)` - Core cleaning function
- `get_cleaning_mask(freq, detector_psds)` - Get boolean cleaning masks
- `apply_cleaning(freq, detector_psds)` - Apply cleaning and return cleaned PSDs

## License

MIT License
