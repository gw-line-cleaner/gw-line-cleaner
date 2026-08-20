# SPDX-FileCopyrightText: 2026 Ye Zhou
#
# SPDX-License-Identifier: MIT

"""Fit baseline to spectrum using iterative polynomial fitting."""

import logging
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from numba import jit, prange
from numpy.polynomial.chebyshev import Chebyshev
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

FitTuple = Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional["Result"]]

logger = logging.getLogger(__name__)

# Silence Numba log messages
logging.getLogger("numba").setLevel(logging.WARNING)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Median filter size for initial baseline estimation
# Larger values produce smoother initial estimates but may miss fine details
FILTER_SIZE = 151

# Minimum prominence required for peak detection in normalized spectrum
# Higher values detect only more significant peaks, lower values detect more peaks
PEAK_PROMINENCE = 3

# Number of points to exclude around each detected peak
# Creates a window of 2*PEAK_WINDOW+1 points around each peak center
PEAK_WINDOW = 5

# Degree of Chebyshev polynomial used for baseline fitting
# Higher degrees can fit more complex baselines but may overfit
POLYNOMIAL_DEGREE = 10

# Maximum number of iterative refinement cycles
# More iterations can improve fit quality but increase computation time
MAX_ITERATIONS = 10

# Threshold for outlier detection in terms of MAD (Median Absolute Deviation)
# Points beyond median + OUTLIER_THRESHOLD * MAD are considered outliers
OUTLIER_THRESHOLD = 3.0

# Column name variations for automatic data format detection
FREQUENCY_COLUMNS = ["frequency", "freq", "f", "Frequency", "FREQ"]
ASD_COLUMNS = ["asd", "ASD", "amplitude", "Amplitude"]
PSD_COLUMNS = ["psd", "PSD", "power", "Power"]

# =============================================================================
# RESULT CLASS - CONTAINER FOR FITTING RESULTS
# =============================================================================


@dataclass
class Result:
    """Container class for baseline fitting results.

    Attributes:
        baseline (np.ndarray): Fitted baseline values corresponding to input frequencies
        rms (float): Root mean square deviation between data and baseline (quality metric)
        time (float): Total processing time in seconds
        iterations (int): Number of iterative refinement cycles completed
        data_used (float): Percentage of original data points used in final fit
        data_type (str): Input data type ('asd' or 'psd')
    """

    baseline: np.ndarray
    rms: float
    time: float
    iterations: int
    data_used: float
    data_type: str

    def plot(
        self,
        frequency: np.ndarray,
        spectrum: np.ndarray,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        detector_name: Optional[str] = None,
    ) -> None:
        """Create a log-log plot comparing original spectrum with fitted baseline.

        Args:
            frequency (np.ndarray): Frequency array (Hz)
            spectrum (np.ndarray): Original spectrum data
            title (Optional[str]): Optional custom plot title
            save_path (Optional[str]): Optional file path to save the plot
            detector_name(Optional[str]):
                Optional detector identifier for labeling (e.g., "H1", "V1")
        """
        plt.figure(figsize=(12, 6))

        # Use detector name or generic label
        spectrum_label = f"{detector_name} Spectrum" if detector_name else "Spectrum"

        # Plot original spectrum with transparency
        plt.loglog(
            frequency, spectrum, "k-", alpha=0.7, linewidth=0.8, label=spectrum_label
        )

        # Plot fitted baseline with thicker line
        plt.loglog(
            frequency,
            self.baseline,
            "r-",
            linewidth=2.5,
            label=f"Baseline Fit (degree {POLYNOMIAL_DEGREE})",
        )

        # Set y-axis limits with some padding for better visualization
        plt.ylim(np.min(spectrum) * 0.5, np.max(spectrum) * 2)
        plt.xlabel("Frequency (Hz)", fontsize=12)
        plt.ylabel("ASD (1/√Hz)", fontsize=12)

        # Generate title with key metrics if not provided
        if title is None:
            prefix = f"{detector_name} " if detector_name else ""
            title = f"{prefix}Baseline Fit (RMS: {self.rms:.4f}, {self.data_type.upper()} data)"
        plt.title(title, fontweight="bold", fontsize=14)

        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def info(self, detector_name: Optional[str] = None) -> None:
        """Print detailed information about the fitting results."""
        prefix = f"{detector_name} " if detector_name else ""
        logger.info(f"{prefix}Baseline Fitting Results")
        logger.info("=" * 40)
        logger.info(f"Polynomial degree: {POLYNOMIAL_DEGREE}")
        logger.info(f"Iterations: {self.iterations}")
        logger.info(f"Data usage: {self.data_used:.1f}% of points")
        logger.info(f"RMS quality: {self.rms:.4f}")
        logger.info(f"Processing time: {self.time:.2f} seconds")
        logger.info(f"Input data type: {self.data_type.upper()}")

    def find_noise_lines(
        self, frequency: np.ndarray, spectrum: np.ndarray, threshold: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Identify potential noise lines by finding points significantly above baseline.

        Args:
            frequency (np.ndarray): Frequency array
            spectrum (np.ndarray): Original spectrum data
            threshold (float): Ratio threshold above baseline to consider as noise line

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Tuple of (noise_frequencies, noise_amplitudes) for detected lines
        """
        ratio = spectrum / self.baseline
        noise_mask = ratio > threshold
        return frequency[noise_mask], spectrum[noise_mask]


# =============================================================================
# JIT-COMPILED OPTIMIZATION FUNCTIONS
# These functions are compiled to machine code for maximum performance
# =============================================================================


@jit(nopython=True, cache=True)
def _fast_normalize_spectrum(
    asd_spectrum: np.ndarray, baseline_estimate: np.ndarray
) -> np.ndarray:
    """Normalize spectrum by dividing by baseline estimate (JIT compiled).

    Args:
        asd_spectrum (np.ndarray): Input spectrum values
        baseline_estimate (np.ndarray): Baseline estimate values

    Returns:
        np.ndarray: Normalized spectrum (ratio of spectrum to baseline)
    """
    normalized = np.empty_like(asd_spectrum)
    for i in prange(len(asd_spectrum)):
        # Avoid division by zero or very small numbers
        if baseline_estimate[i] > 1e-30:
            normalized[i] = asd_spectrum[i] / baseline_estimate[i]
        else:
            normalized[i] = 1.0
    return normalized


@jit(nopython=True, cache=True)
def _fast_remove_peaks(
    keep_mask: np.ndarray, peak_indices: np.ndarray, peak_window: int, n_points: int
) -> np.ndarray:
    """Mark points around detected peaks for exclusion (JIT compiled).

    Args:
        keep_mask (np.ndarray): Boolean array indicating which points to keep
        peak_indices (np.ndarray): Array of peak center indices
        peak_window (int): Number of points to exclude on each side of peak
        n_points (int): Total number of points in dataset

    Returns:
        np.ndarray: Updated keep_mask with peak regions marked as False
    """
    for i in prange(len(peak_indices)):
        peak_idx = peak_indices[i]
        # Define exclusion window around peak, respecting array boundaries
        start_idx = max(0, peak_idx - peak_window)
        end_idx = min(n_points, peak_idx + peak_window + 1)
        # Mark all points in window for exclusion
        for j in range(start_idx, end_idx):
            keep_mask[j] = False
    return keep_mask


@jit(nopython=True, cache=True)
def _fast_outlier_detection(
    log_residuals: np.ndarray, median_residual: np.ndarray, mad: float, threshold: float
) -> np.ndarray:
    """Detect outliers using robust statistics (JIT compiled).

    Args:
        log_residuals (np.ndarray): Log-scale residuals (log of ratio between data and fit)
        median_residual (np.ndarray): Median of residuals (robust center estimate)
        mad (float): Median Absolute Deviation (robust scale estimate)
        threshold (float): Number of MAD units beyond median to consider outlier

    Returns:
        np.ndarray: Boolean array marking outliers as True
    """
    outlier_mask = np.empty(len(log_residuals), dtype=np.bool8)
    cutoff = median_residual + threshold * mad
    for i in prange(len(log_residuals)):
        outlier_mask[i] = log_residuals[i] > cutoff
    return outlier_mask


@jit(nopython=True, cache=True)
def _fast_log_safe(data: np.ndarray) -> np.ndarray:
    """Compute base-10 logarithm safely, avoiding log of zero or negative (JIT compiled).

    Args:
        data (np.ndarray): Input array

    Returns:
        np.ndarray: Log10 of data, with very small/negative values set to 0
    """
    out = np.empty_like(data)
    for i in prange(len(data)):
        if data[i] > 1e-30:
            # Manual log10 calculation for numba compatibility
            out[i] = np.log(data[i]) / np.log(10.0)
        else:
            out[i] = 0.0
    return out


@jit(nopython=True, cache=True)
def _fast_ratio_calculation(
    asd_spectrum: np.ndarray, baseline: np.ndarray
) -> np.ndarray:
    """Calculate ratio between spectrum and baseline safely (JIT compiled).

    Args:
        asd_spectrum (np.ndarray): Spectrum values
        baseline (np.ndarray): Baseline values

    Returns:
        np.ndarray: Ratio array with protection against division by zero
    """
    out = np.empty_like(asd_spectrum)
    for i in prange(len(asd_spectrum)):
        if baseline[i] > 1e-30:
            out[i] = asd_spectrum[i] / baseline[i]
        else:
            out[i] = 1.0
    return out


@jit(nopython=True, cache=True)
def _fast_sqrt(data: np.ndarray) -> np.ndarray:
    """Compute square root safely for JIT compatibility.

    Args:
        data (np.ndarray): Input array

    Returns:
        np.ndarray: Square root of data, with negative values set to 0
    """
    out = np.empty_like(data)
    for i in prange(len(data)):
        if data[i] > 0:
            out[i] = np.sqrt(data[i])
        else:
            out[i] = 0.0
    return out


# =============================================================================
# DATA LOADING AND VALIDATION FUNCTIONS
# =============================================================================


def load(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load spectral data from file and convert to ASD format.

    Supports both ECSV (Enhanced Character Separated Values) and plain text formats.
    Automatically detects data type (ASD vs PSD) and converts PSD to ASD if needed.

    Args:
        file_path (str): Path to input data file

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (frequency_array, asd_spectrum_array)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or data is corrupted
    """
    if not os.path.exists(file_path):
        msg = f"Spectrum file not found: {file_path}"
        raise FileNotFoundError(msg)

    # Determine file format from extension
    _, ext = os.path.splitext(file_path.lower())

    try:
        if ext == ".ecsv":
            frequency, data, data_type = _load_ecsv(file_path)
        elif ext in [".txt", ".dat"]:
            frequency, data, data_type = _load_txt(file_path)
        else:
            # Try ECSV first, fall back to text format
            try:
                frequency, data, data_type = _load_ecsv(file_path)
            except Exception:
                frequency, data, data_type = _load_txt(file_path)
    except Exception as e:
        msg = f"Failed to load data from {file_path}: {str(e)}"
        raise ValueError(msg)

    # Convert PSD to ASD if necessary (ASD = sqrt(PSD))
    if data_type == "psd":
        asd_spectrum = _fast_sqrt(data)
    else:
        asd_spectrum = data.copy()

    # Validate loaded data for consistency and physical reasonableness
    _validate_data(frequency, asd_spectrum)

    return frequency, asd_spectrum


def _load_ecsv(file_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load data from ECSV (Enhanced Character Separated Values) format.

    ECSV is an ASCII table format used by Astropy with metadata support.
    Automatically detects column names for frequency and spectrum data.

    Args:
        file_path (str): Path to ECSV file

    Returns:
        Tuple[np.ndarray, np.ndarray, str]: Tuple of (frequency, data_values, data_type)
    """
    data_table = Table.read(file_path)

    if len(data_table) == 0:
        msg = "ECSV file contains no data"
        raise ValueError(msg)

    # Search for frequency column using common naming conventions
    freq_col = None
    for col in FREQUENCY_COLUMNS:
        if col in data_table.colnames:
            freq_col = col
            break

    # Fall back to first column if no standard name found
    if freq_col is None:
        freq_col = data_table.colnames[0]

    # Convert to contiguous array for better performance
    frequency = np.ascontiguousarray(data_table[freq_col], dtype=np.float64)

    # Search for spectrum data column and determine if it's ASD or PSD
    data_col = None
    data_type = None

    # First try to find ASD columns
    for col in ASD_COLUMNS:
        if col in data_table.colnames:
            data_col = col
            data_type = "asd"
            break

    # If no ASD column found, try PSD columns
    if data_col is None:
        for col in PSD_COLUMNS:
            if col in data_table.colnames:
                data_col = col
                data_type = "psd"
                break

    # Fall back to second column and guess data type
    if data_col is None:
        data_col = data_table.colnames[1]
        # Use first 100 points to guess data type based on magnitude
        data_type = _guess_data_type(np.array(data_table[data_col][:100]))
    assert data_type is not None

    data_values = np.ascontiguousarray(data_table[data_col], dtype=np.float64)

    # Filter out invalid data (negative, zero, or non-finite values)
    valid_mask = (
        (frequency > 0)
        & (data_values > 0)
        & np.isfinite(frequency)
        & np.isfinite(data_values)
    )
    return frequency[valid_mask], data_values[valid_mask], data_type


def _load_txt(file_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load data from plain text format file.

    Assumes space/tab separated columns with frequency in first column
    and spectrum data in second column. Ignores comment lines starting with % or #.

    Args:
        file_path (str): Path to text file

    Returns:
        Tuple[np.ndarray, np.ndarray, str]: Tuple of (frequency, data_values, data_type)
    """
    # Load data skipping comment lines
    data_array = np.loadtxt(file_path, comments=["%", "#"], dtype=np.float64)

    if data_array.size == 0:
        msg = "Text file contains no data"
        raise ValueError(msg)

    # Handle case of single row (reshape for consistent indexing)
    if data_array.ndim == 1:
        data_array = data_array.reshape(1, -1)

    if data_array.shape[1] < 2:
        msg = "Text file must have at least 2 columns"
        raise ValueError(msg)

    # Extract frequency and data columns
    frequency = data_array[:, 0].copy()
    data_values = data_array[:, 1].copy()

    # Guess data type from magnitude (sample first 100 points for efficiency)
    sample_data = data_values[:100] if len(data_values) > 100 else data_values
    data_type = _guess_data_type(sample_data)

    # Filter out invalid data points
    valid_mask = (
        (frequency > 0)
        & (data_values > 0)
        & np.isfinite(frequency)
        & np.isfinite(data_values)
    )
    return frequency[valid_mask], data_values[valid_mask], data_type


def _guess_data_type(data_values: np.ndarray) -> str:
    """Guess whether data represents ASD or PSD based on typical magnitude ranges.

    ASD values are typically in range 1e-24 to 1e-18 (1/√Hz)
    PSD values are typically much smaller due to being squared

    Args:
        data_values (np.ndarray): Sample of data values to analyze

    Returns:
        str: String 'asd' or 'psd' indicating guessed data type
    """
    if len(data_values) == 0:
        return "asd"  # Default assumption

    # Use median for robust estimation against outliers
    median_val = np.median(data_values)
    # Threshold based on typical spectrum magnitudes
    return "psd" if median_val < 1e-35 else "asd"


def _validate_data(frequency: np.ndarray, asd_spectrum: np.ndarray) -> None:
    """Validate input data arrays for physical consistency and numerical stability.

    Args:
        frequency (np.ndarray): Frequency array (Hz)
        asd_spectrum (np.ndarray): ASD spectrum array (1/√Hz)

    Raises:
        ValueError: If data fails validation checks
    """
    # Check array length consistency
    if len(frequency) != len(asd_spectrum):
        msg = "Frequency and spectrum arrays have different lengths"
        raise ValueError(msg)

    # Ensure sufficient data points for reliable fitting
    if len(frequency) < 50:
        msg = f"Insufficient data points: {len(frequency)} (minimum 50 required)"
        raise ValueError(msg)

    # Check for physical validity (all positive values)
    if not np.all(frequency > 0):
        msg = "All frequency values must be positive"
        raise ValueError(msg)

    if not np.all(asd_spectrum > 0):
        msg = "All spectrum values must be positive"
        raise ValueError(msg)

    # Check for numerical validity (no infinite or NaN values)
    if not np.all(np.isfinite(frequency)):
        msg = "Frequency array contains infinite or NaN values"
        raise ValueError(msg)

    if not np.all(np.isfinite(asd_spectrum)):
        msg = "Spectrum array contains infinite or NaN values"
        raise ValueError(msg)


# =============================================================================
# CORE FITTING ALGORITHM
# =============================================================================


def fit(frequency: np.ndarray, asd_spectrum: np.ndarray) -> Result:
    """Fit baseline to spectrum using iterative polynomial fitting.

    Algorithm overview:
    1. Initial baseline estimation using median filtering
    2. Peak detection and removal in normalized spectrum
    3. Polynomial fitting in log-log space
    4. Iterative refinement with outlier removal
    5. Quality assessment and result packaging

    Args:
        frequency (np.ndarray): Frequency array (Hz) - must be positive and sorted
        asd_spectrum (np.ndarray): ASD spectrum array (1/√Hz) - must be positive

    Returns:
        Result: Result object containing fitted baseline and quality metrics

    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If fitting process fails
    """
    start_time = time.time()

    # Ensure inputs are proper numpy arrays with optimal memory layout
    if not isinstance(frequency, np.ndarray) or not isinstance(
        asd_spectrum, np.ndarray
    ):
        msg = "Inputs must be numpy arrays"
        raise TypeError(msg)

    frequency = np.ascontiguousarray(frequency, dtype=np.float64)
    asd_spectrum = np.ascontiguousarray(asd_spectrum, dtype=np.float64)

    n_points = len(frequency)

    # Warn about potentially unreliable results for small datasets
    if n_points < 100:
        warnings.warn(
            f"Small dataset ({n_points} points) may produce unreliable results"
        )

    # Step 1: Initial baseline estimation using median filtering
    # Median filter removes narrow features while preserving broad spectral shape
    filter_size = min(FILTER_SIZE, n_points // 3)  # Adapt filter size to data length
    baseline_estimate = median_filter(asd_spectrum, size=filter_size)

    # Step 2: Normalize spectrum by initial baseline and detect peaks
    # Normalization makes peak detection threshold-independent of absolute scale
    normalized_spectrum = _fast_normalize_spectrum(asd_spectrum, baseline_estimate)

    try:
        # Find peaks that stand out significantly above the baseline
        peak_indices, _ = find_peaks(normalized_spectrum, prominence=PEAK_PROMINENCE)
    except Exception as e:
        warnings.warn(f"Peak detection failed: {str(e)}")
        peak_indices = np.array([], dtype=int)

    # Step 3: Remove detected peaks and surrounding points from fitting data
    # This prevents narrow spectral lines from biasing the baseline fit
    keep_mask = np.ones(n_points, dtype=bool)
    if len(peak_indices) > 0:
        keep_mask = _fast_remove_peaks(keep_mask, peak_indices, PEAK_WINDOW, n_points)

    clean_frequency = frequency[keep_mask]
    clean_spectrum = asd_spectrum[keep_mask]

    if len(clean_frequency) < 20:
        warnings.warn(
            "Very few points remaining after peak removal. Results may be unreliable."
        )

    # Step 4: Initial polynomial fit using peak-cleaned data
    current_baseline = _fit_polynomial(clean_frequency, clean_spectrum, frequency)

    # Step 5: Iterative refinement with outlier detection and removal
    iterations_completed = 1
    refined_frequency = clean_frequency
    refined_spectrum = clean_spectrum

    for iteration in range(1, MAX_ITERATIONS):
        try:
            # Calculate residuals between data and current fit
            ratio_array = _fast_ratio_calculation(asd_spectrum, current_baseline)
            log_residuals = _fast_log_safe(ratio_array)

            # Use robust statistics for outlier detection
            # Median and MAD are less sensitive to outliers than mean and std
            median_residual = np.median(log_residuals)
            mad = (
                np.median(np.abs(log_residuals - median_residual)) * 1.4826
            )  # Scale factor for normal distribution

            # Check for convergence (very small scatter indicates good fit)
            if mad < 1e-10:
                break

            # Identify and remove outliers
            outlier_mask = _fast_outlier_detection(
                log_residuals, median_residual, mad, OUTLIER_THRESHOLD
            )

            refined_frequency = frequency[~outlier_mask]
            refined_spectrum = asd_spectrum[~outlier_mask]

            # Stop if too much data has been excluded (may indicate poor fit)
            if len(refined_frequency) / n_points < 0.3:
                break

            # Refit polynomial with outliers removed
            current_baseline = _fit_polynomial(
                refined_frequency, refined_spectrum, frequency
            )
            iterations_completed = iteration + 1

        except Exception:
            break

    # Step 6: Calculate final quality metrics and package results
    final_ratio = asd_spectrum / current_baseline
    # RMS of deviation from unity indicates overall fit quality
    rms_quality = np.sqrt(np.mean((final_ratio - 1.0) ** 2))
    data_usage = 100.0 * len(refined_frequency) / n_points
    processing_time = time.time() - start_time

    return Result(
        baseline=current_baseline,
        rms=rms_quality,
        time=processing_time,
        iterations=iterations_completed,
        data_used=data_usage,
        data_type="asd",
    )


def _fit_polynomial(
    clean_frequency: np.ndarray, clean_spectrum: np.ndarray, full_frequency: np.ndarray
) -> np.ndarray:
    """Fit Chebyshev polynomial to spectrum data in log-log space.

    Chebyshev polynomials provide better numerical stability than standard
    polynomials for high-degree fits. Working in log space linearizes
    the typically power-law behavior of noise spectra.

    Args:
        clean_frequency (np.ndarray): Frequency points to fit (outliers/peaks removed)
        clean_spectrum (np.ndarray): Spectrum values to fit (outliers/peaks removed)
        full_frequency (np.ndarray): Full frequency grid for baseline evaluation

    Returns:
        np.ndarray: Baseline values evaluated at full_frequency points

    Raises:
        ValueError: If insufficient points for polynomial degree
    """
    if len(clean_frequency) < POLYNOMIAL_DEGREE + 1:
        msg = f"Insufficient points for degree {POLYNOMIAL_DEGREE} polynomial: {len(clean_frequency)}"
        raise ValueError(msg)

    try:
        # Convert to log scale for fitting (log-log relationship is common in spectra)
        log_freq_clean = _fast_log_safe(clean_frequency)
        log_spec_clean = _fast_log_safe(clean_spectrum)
        log_freq_full = _fast_log_safe(full_frequency)

        # Fit Chebyshev polynomial (numerically stable for high degrees)
        polynomial = Chebyshev.fit(log_freq_clean, log_spec_clean, POLYNOMIAL_DEGREE)

        # Evaluate polynomial at all frequency points
        log_baseline = polynomial(log_freq_full)

        # Convert back to linear scale
        baseline = np.power(10.0, log_baseline)

        return baseline

    except Exception as e:
        msg = f"Polynomial fitting failed: {str(e)}"
        raise ValueError(msg)


# =============================================================================
# CONVENIENCE FUNCTIONS FOR USER INTERFACE
# =============================================================================


def load_and_fit(detector_asd_src: Union[str, np.ndarray]) -> FitTuple:
    """Convenience function to load data and fit baseline in one step.

    Args:
        detector_asd_src (Union[str, np.ndarray]):
            Either
            - a file supported by load() containing the spectrum data, or
            - a NumPy array of the spectrum.
            In either case, columns are (frequency, asd_spectrum).

    Returns:
        FitTuple: Tuple of (frequency, asd_spectrum, result)

    Raises:
        ValueError: If loading or fitting fails
    """
    if isinstance(detector_asd_src, np.ndarray):
        detector_asd_src_str = np.array2string(detector_asd_src, threshold=5)
    else:
        detector_asd_src_str = detector_asd_src
    try:
        if isinstance(detector_asd_src, np.ndarray):
            frequency = detector_asd_src[:, 0]
            asd_spectrum = detector_asd_src[:, 1]
        else:
            frequency, asd_spectrum = load(detector_asd_src)
        result = fit(frequency, asd_spectrum)
        return frequency, asd_spectrum, result
    except Exception as e:
        msg = f"Failed to load and fit {detector_asd_src_str}: {str(e)}"
        raise ValueError(msg)


def load_and_fit_multiple(
    detector_asd_srcs: Dict[str, Union[str, np.ndarray]],
) -> Dict[str, FitTuple]:
    """Convenience function to load and fit multiple detector files.

    Args:
        detector_asd_srcs (Dict[str, Union[str, np.ndarray]]):
            Dictionary mapping detector names to file paths containing ASDs:
                e.g., {"H1": "/path/to/h1.txt", "L1": "/path/to/l1.txt", "V1": "/path/to/v1.txt"}
            or mapping detector names to NumPy arrays containing ASDs:
                e.g., {"H1": np.array(...), "L1": np.array(...), "V1": np.array(...)}

    Returns:
        Dict[str, FitTuple]:
            Dictionary mapping detector names to (frequency, asd_spectrum, result) tuples
    """
    results = {}
    for det_name, detector_asd_src in detector_asd_srcs.items():
        frequency, asd_spectrum, result = load_and_fit(detector_asd_src)
        results[det_name] = (frequency, asd_spectrum, result)
    return results


def batch(file_list: List[str]) -> List[FitTuple]:
    """Process multiple spectrum files in batch mode.

    Attempts to process each file independently, continuing even if some fail.
    Provides summary statistics of success/failure rates.

    Args:
        file_list (List[str]): List of file paths to process

    Returns:
        List[FitTuple]:
            List of tuples (frequency, asd_spectrum, result) for each file
            Failed files have (None, None, None) entries
    """
    if not file_list:
        return []

    results = []
    successful = 0

    for i, file_path in enumerate(file_list):
        try:
            frequency, asd_spectrum, result = load_and_fit(file_path)
            results.append((frequency, asd_spectrum, result))
            successful += 1
        except Exception:
            results.append((None, None, None))

    return results


# =============================================================================
# PUBLIC API DEFINITION
# =============================================================================

__all__ = ["load", "fit", "load_and_fit", "load_and_fit_multiple", "batch", "Result"]
