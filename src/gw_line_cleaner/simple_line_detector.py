import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import time
import warnings
from . import baseline_fit

# =============================================================================
# DETECTION PARAMETERS - Control sensitivity and behavior of line detection
# =============================================================================

# Minimum peak amplitude relative to baseline (e.g., 2.5 = 150% above baseline)
GLOBAL_MIN_AMPLITUDE = 2.5

# Minimum prominence for peak detection (difference between peak and surrounding valleys)
GLOBAL_MIN_PROMINENCE = 0.1

# Baseline level threshold for determining line boundaries (e.g., 1.05 = 5% above baseline)
GLOBAL_BASELINE_LEVEL = 1.05

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Line:
    """
    Container for spectral line properties
    
    Attributes:
        frequency: Peak frequency in Hz
        amplitude: Peak amplitude relative to baseline
        left_boundary: Left frequency boundary where line returns to baseline
        right_boundary: Right frequency boundary where line returns to baseline
        width: Frequency width of the line (right_boundary - left_boundary)
        detector: Optional detector identifier (e.g., "H1", "L1", "V1")
    """
    frequency: float
    amplitude: float
    left_boundary: float
    right_boundary: float
    width: float
    detector: Optional[str] = None
    
    def __post_init__(self):
        """Validate line data for physical consistency after initialization"""
        if self.width <= 0:
            raise ValueError(f"Line width must be positive: {self.width}")
        if self.frequency <= 0:
            raise ValueError(f"Frequency must be positive: {self.frequency}")
        if self.amplitude <= 0:
            raise ValueError(f"Amplitude must be positive: {self.amplitude}")

# =============================================================================
# MAIN DETECTION ALGORITHM
# =============================================================================

def detect_lines(frequency: np.ndarray, spectrum: np.ndarray, baseline: np.ndarray = None,
                min_amplitude: float = GLOBAL_MIN_AMPLITUDE, 
                baseline_level: float = GLOBAL_BASELINE_LEVEL,
                min_prominence: float = GLOBAL_MIN_PROMINENCE,
                detector_name: Optional[str] = None) -> List[Line]:
    """
    Detect spectral lines in data using prominence-based peak detection
    
    Algorithm overview:
    1. Use baseline_fit to obtain baseline if not provided
    2. Validate input data for consistency
    3. Calculate spectrum-to-baseline ratio
    4. Find peaks using scipy's find_peaks with amplitude and prominence filters
    5. Determine physical boundaries for each peak
    6. Merge nearby lines using logarithmic frequency-dependent clustering
    7. Return sorted list of unique spectral lines
    
    Args:
        frequency: Frequency array in Hz (must be positive and monotonic)
        spectrum: Measured spectrum values (ASD in 1/√Hz)
        baseline: Fitted baseline values (optional - will use baseline_fit if not provided)
        min_amplitude: Minimum peak height relative to baseline (default 2.5)
        baseline_level: Threshold for boundary detection (default 1.05 = 5% above baseline)
        min_prominence: Minimum prominence for peak detection (default 0.10)
        detector_name: Optional detector identifier to attach to detected lines
        
    Returns:
        List of Line objects sorted by frequency
        
    Raises:
        ValueError: If input arrays have inconsistent sizes or invalid values
    """
    
    # Use baseline_fit to obtain baseline if not provided
    if baseline is None:
        baseline = baseline_fit.fit(frequency, spectrum).baseline
    
    # Validate input array dimensions and basic properties
    if len(frequency) != len(spectrum) or len(spectrum) != len(baseline):
        raise ValueError("Frequency, spectrum, and baseline arrays must have identical lengths")
    
    if len(frequency) < 10:
        raise ValueError("Need at least 10 data points for reliable line detection")
    
    if not np.all(frequency > 0):
        raise ValueError("All frequency values must be positive")
    
    if not np.all(spectrum > 0) or not np.all(baseline > 0):
        raise ValueError("All spectrum and baseline values must be positive")
    
    # Calculate normalized spectrum (ratio to baseline)
    # This removes the overall spectral shape and highlights deviations
    ratio = np.divide(spectrum, baseline, out=np.ones_like(spectrum), where=baseline > 1e-30)
    
    # Peak detection using scipy's find_peaks algorithm
    peak_indices, properties = find_peaks(
        ratio, 
        height=min_amplitude,      # Minimum peak height
        prominence=min_prominence, # Minimum prominence above surroundings
        distance=2,               # Minimum separation between peaks (data points)
        width=0                   # Minimum width (0 = any width allowed)
    )
    
    if len(peak_indices) == 0:
        return []
    
    # Find physical boundaries for each detected peak
    candidate_lines = []
    for peak_idx in peak_indices:
        line = _find_physical_boundaries(frequency, ratio, peak_idx, baseline_level, detector_name)
        if line is not None:
            candidate_lines.append(line)
    
    if not candidate_lines:
        return []
    
    # Apply intelligent clustering to merge nearby or overlapping lines
    unique_lines = _logarithmic_clustering(candidate_lines)
    
    # Return lines sorted by frequency for consistent output
    return sorted(unique_lines, key=lambda x: x.frequency)


def detect_lines_multiple(
    detector_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    **kwargs
) -> Dict[str, List[Line]]:
    """
    Detect lines for multiple detectors.
    
    Args:
        detector_data: Dictionary mapping detector names to 
                       (frequency, spectrum, baseline) tuples
        **kwargs: Additional arguments passed to detect_lines
    
    Returns:
        Dictionary mapping detector names to lists of detected Lines
    """
    results = {}
    for det_name, (freq, spectrum, baseline) in detector_data.items():
        results[det_name] = detect_lines(freq, spectrum, baseline, 
                                         detector_name=det_name, **kwargs)
    return results

# =============================================================================
# BOUNDARY DETECTION FUNCTIONS
# =============================================================================

def _find_physical_boundaries(frequency: np.ndarray, ratio: np.ndarray, 
                             peak_idx: int, baseline_level: float,
                             detector_name: Optional[str] = None) -> Optional[Line]:
    """
    Determine the frequency boundaries where a spectral line begins and ends
    
    Algorithm:
    1. Start from peak center and search outward in both directions
    2. Find points where ratio drops to baseline_level (e.g., 5% above baseline)
    3. Handle edge cases where line extends to data boundaries
    4. Validate that boundaries make physical sense
    
    Args:
        frequency: Frequency array
        ratio: Spectrum-to-baseline ratio array
        peak_idx: Index of peak center in the arrays
        baseline_level: Threshold ratio for determining line boundaries
        detector_name: Optional detector identifier
        
    Returns:
        Line object if valid boundaries found, None otherwise
    """
    
    n = len(ratio)
    
    # Validate peak index is within array bounds
    if peak_idx < 0 or peak_idx >= n:
        return None
    
    peak_freq = frequency[peak_idx]
    peak_amp = ratio[peak_idx]
    
    # Validate peak amplitude meets minimum requirements
    if peak_amp <= baseline_level or not np.isfinite(peak_amp):
        return None
    
    # Search for left boundary by moving backward from peak
    left_idx = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        # Stop when ratio drops to baseline level or encounters invalid data
        if not np.isfinite(ratio[i]) or ratio[i] <= baseline_level:
            left_idx = i
            break
        left_idx = i  # Continue search if still above baseline
    
    # Search for right boundary by moving forward from peak
    right_idx = peak_idx
    for i in range(peak_idx + 1, n):
        # Stop when ratio drops to baseline level or encounters invalid data
        if not np.isfinite(ratio[i]) or ratio[i] <= baseline_level:
            right_idx = i
            break
        right_idx = i  # Continue search if still above baseline
    
    # Validate that boundaries form a reasonable line
    if left_idx >= right_idx:
        return None
    
    left_freq = frequency[left_idx]
    right_freq = frequency[right_idx]
    width = right_freq - left_freq
    
    # Check for valid width calculation
    if width <= 0 or not np.isfinite(width):
        return None
    
    # Create and return Line object with validation
    try:
        return Line(
            frequency=peak_freq,
            amplitude=peak_amp,
            left_boundary=left_freq,
            right_boundary=right_freq,
            width=width,
            detector=detector_name
        )
    except ValueError:
        return None

# =============================================================================
# INTELLIGENT CLUSTERING ALGORITHMS
# =============================================================================

def _get_frequency_threshold(frequency: float) -> float:
    """
    Calculate frequency-dependent threshold for merging nearby lines
    
    Uses logarithmic scaling to account for the fact that frequency resolution
    requirements change across the band. At low frequencies, lines can be
    closer together, while at high frequencies, wider separation is needed.
    
    Args:
        frequency: Reference frequency in Hz
        
    Returns:
        Frequency ratio threshold (e.g., 1.02 = 2% frequency difference)
    """
    
    if frequency <= 0:
        return 1.02  # Default 2% threshold for invalid input
    
    log_freq = np.log10(frequency)
    
    # Reference calibration: 2% threshold at 10 Hz
    log_reference = 1.0  # log10(10) = 1.0
    base_threshold = 0.02  # 2% relative frequency difference
    
    # Logarithmic scaling: threshold increases with frequency
    # Reaches approximately 7% at 2000 Hz
    scaling_slope = 0.022
    
    # Calculate frequency-dependent relative threshold
    relative_threshold = base_threshold + scaling_slope * (log_freq - log_reference)
    
    # Apply reasonable limits: 1.5% minimum, 10% maximum
    relative_threshold = np.clip(relative_threshold, 0.015, 0.10)
    
    # Convert to frequency ratio (1 + relative difference)
    frequency_ratio_threshold = 1.0 + relative_threshold
    
    return frequency_ratio_threshold

def _get_gap_threshold(frequency: float) -> float:
    """
    Calculate frequency-dependent threshold for maximum allowed gap between lines
    
    Determines how large a frequency gap can exist between two lines before
    they are considered separate features rather than parts of the same line.
    
    Args:
        frequency: Reference frequency in Hz
        
    Returns:
        Maximum relative gap as fraction of frequency (e.g., 0.005 = 0.5%)
    """
    
    if frequency <= 0:
        return 0.005  # Default 0.5% gap threshold
    
    log_freq = np.log10(frequency)
    
    # Base gap threshold at reference frequency
    base_gap = 0.005  # 0.5% of frequency
    scaling_factor = 0.015  # Growth rate with log frequency
    
    # Calculate frequency-dependent gap threshold
    relative_gap = base_gap + scaling_factor * (log_freq - 1.0)
    
    # Apply reasonable bounds: 0.5% minimum, 5% maximum
    relative_gap = np.clip(relative_gap, 0.005, 0.05)
    
    return relative_gap

def _should_merge_lines(line1: Line, line2: Line) -> bool:
    """
    Determine whether two spectral lines should be merged into a single line
    
    Uses multiple criteria:
    1. Boundary overlap: Lines with overlapping frequency ranges are always merged
    2. Frequency similarity: Lines with similar frequencies (within threshold) are candidates
    3. Gap analysis: Small gaps between lines suggest they're part of same feature
    
    Args:
        line1, line2: Line objects to compare
        
    Returns:
        True if lines should be merged, False otherwise
    """
    
    freq1, freq2 = line1.frequency, line2.frequency
    
    # Criterion 1: Check for boundary overlap (mandatory merge condition)
    overlap_start = max(line1.left_boundary, line2.left_boundary)
    overlap_end = min(line1.right_boundary, line2.right_boundary)
    if overlap_start < overlap_end:
        return True  # Overlapping lines must be merged
    
    # Criterion 2: Frequency and gap analysis for nearby lines
    if freq1 <= freq2:
        lower_freq, higher_freq = freq1, freq2
        left_line, right_line = line1, line2
    else:
        lower_freq, higher_freq = freq2, freq1
        left_line, right_line = line2, line1
    
    # Calculate frequency ratio with safety check
    if lower_freq <= 0:
        return False
    
    freq_ratio = higher_freq / lower_freq
    
    # Use average frequency for threshold calculations
    avg_freq = (lower_freq + higher_freq) / 2
    freq_threshold = _get_frequency_threshold(avg_freq)
    
    # Calculate gap between line boundaries
    gap = right_line.left_boundary - left_line.right_boundary
    gap_threshold = _get_gap_threshold(avg_freq)
    max_allowed_gap = lower_freq * gap_threshold
    
    # Apply both frequency and gap criteria
    freq_similar = freq_ratio <= freq_threshold
    gap_acceptable = gap <= max_allowed_gap
    
    return freq_similar and gap_acceptable

def _logarithmic_clustering(lines: List[Line]) -> List[Line]:
    """
    Apply intelligent clustering to merge related spectral lines
    
    Algorithm:
    1. Sort lines by amplitude (strongest first) to prioritize dominant features
    2. For each line, check if it should merge with any already-selected line
    3. If mergeable, combine with existing line; otherwise add as new line
    4. This preserves the strongest features while merging weak satellites
    
    Args:
        lines: List of candidate Line objects
        
    Returns:
        List of unique Line objects after clustering
    """
    
    if len(lines) <= 1:
        return lines
    
    # Sort by amplitude (strongest peaks processed first)
    # This ensures dominant features are preserved during merging
    sorted_by_amplitude = sorted(lines, key=lambda x: x.amplitude, reverse=True)
    
    clustered_lines = []
    
    for candidate in sorted_by_amplitude:
        merged_with_existing = False
        
        # Check if candidate should merge with any existing clustered line
        for i, existing_line in enumerate(clustered_lines):
            if _should_merge_lines(candidate, existing_line):
                # Merge candidate into existing line
                merged_line = _merge_two_lines(candidate, existing_line)
                clustered_lines[i] = merged_line
                merged_with_existing = True
                break
        
        # If no merger occurred, add as new independent line
        if not merged_with_existing:
            clustered_lines.append(candidate)
    
    return clustered_lines

def _merge_two_lines(line1: Line, line2: Line) -> Line:
    """
    Combine two spectral lines into a single merged line
    
    Strategy:
    1. Use properties of the stronger (higher amplitude) line as primary
    2. Expand boundaries to encompass both original lines
    3. Validate merged line properties for physical consistency
    
    Args:
        line1, line2: Line objects to merge
        
    Returns:
        New Line object representing the merged result
        
    Raises:
        ValueError: If merge produces invalid line properties
    """
    
    # Choose dominant line based on amplitude
    if line1.amplitude >= line2.amplitude:
        dominant_line = line1
    else:
        dominant_line = line2
    
    # Expand boundaries to encompass both lines
    merged_left = min(line1.left_boundary, line2.left_boundary)
    merged_right = max(line1.right_boundary, line2.right_boundary)
    
    merged_width = merged_right - merged_left
    
    # Validate merged properties
    if merged_width <= 0:
        raise ValueError(f"Merged line has invalid width: {merged_width}")
    
    return Line(
        frequency=dominant_line.frequency,      # Use dominant line's peak frequency
        amplitude=dominant_line.amplitude,      # Use dominant line's amplitude
        left_boundary=merged_left,             # Extended left boundary
        right_boundary=merged_right,           # Extended right boundary
        width=merged_width,                    # Recalculated width
        detector=dominant_line.detector        # Preserve detector info
    )

# =============================================================================
# VISUALIZATION AND OUTPUT FUNCTIONS
# =============================================================================

def plot_result(frequency: np.ndarray, spectrum: np.ndarray, baseline: np.ndarray, 
                lines: List[Line], title: str = None,
                save_path: Optional[str] = None,
                detector_name: Optional[str] = None):
    """
    Create visualization of spectral line detection results
    
    Produces a log-log plot showing:
    - Original spectrum in black
    - Fitted baseline in red
    - Detected spectral lines as orange triangles
    - Shaded regions showing line boundaries
    
    Args:
        frequency: Frequency array (Hz)
        spectrum: Original spectrum data
        baseline: Fitted baseline
        lines: List of detected Line objects
        title: Plot title (auto-generated if None)
        save_path: Optional file path to save plot
        detector_name: Optional detector identifier for labeling
    """
    
    plt.figure(figsize=(12, 6))
    
    # Use detector name or generic label
    spectrum_label = f'{detector_name} Spectrum' if detector_name else 'Spectrum'
    
    # Plot original spectrum and baseline
    plt.loglog(frequency, spectrum, 'k-', alpha=0.7, linewidth=0.8, 
              label=spectrum_label)
    plt.loglog(frequency, baseline, 'r-', linewidth=2.5, 
              label='Baseline Fit')
    
    # Plot detected lines if any exist
    if lines:
        # Extract line frequencies and corresponding spectrum amplitudes
        line_frequencies = [line.frequency for line in lines]
        line_amplitudes = []
        
        # Find spectrum values at line frequencies
        for freq in line_frequencies:
            closest_idx = np.argmin(np.abs(frequency - freq))
            line_amplitudes.append(spectrum[closest_idx])
        
        # Plot line markers
        plt.loglog(line_frequencies, line_amplitudes, '^', 
                  color='orange', markersize=8, alpha=0.8, 
                  label=f'Detected Lines ({len(lines)})')
        
        # Show line boundaries as shaded regions
        for line in lines:
            plt.axvspan(line.left_boundary, line.right_boundary, 
                       alpha=0.15, color='orange')
    
    # Configure plot appearance
    plt.ylim(np.min(spectrum) * 0.5, np.max(spectrum) * 2)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("ASD (1/√Hz)", fontsize=12)
    
    # Auto-generate title if not provided
    if title is None:
        prefix = f"{detector_name} " if detector_name else ""
        title = f"{prefix}Spectral Line Detection"
    plt.title(title, fontweight='bold', fontsize=14)
    
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# =============================================================================
# HIGH-LEVEL USER INTERFACE FUNCTIONS
# =============================================================================

def find_and_plot(frequency: np.ndarray, spectrum: np.ndarray, baseline: np.ndarray = None,
                  min_amplitude: float = GLOBAL_MIN_AMPLITUDE, 
                  baseline_level: float = GLOBAL_BASELINE_LEVEL,
                  min_prominence: float = GLOBAL_MIN_PROMINENCE,
                  save_path: Optional[str] = None,
                  detector_name: Optional[str] = None) -> List[Line]:
    """
    Complete workflow: detect spectral lines and create visualization
    
    Combines line detection with automatic plotting and summary output.
    Provides user-friendly interface for typical use cases.
    
    Args:
        frequency: Frequency array (Hz)
        spectrum: Spectrum data (ASD)
        baseline: Fitted baseline (optional - will use baseline_fit if not provided)
        min_amplitude: Minimum peak amplitude threshold
        baseline_level: Baseline crossing threshold
        min_prominence: Minimum peak prominence
        save_path: Optional path to save plot
        detector_name: Optional detector identifier for labeling
        
    Returns:
        List of detected Line objects
    """
    
    # Perform line detection
    lines = detect_lines(frequency, spectrum, baseline, 
                        min_amplitude, baseline_level, min_prominence,
                        detector_name=detector_name)
    
    # Use provided baseline or fit new one for plotting
    if baseline is None:
        baseline = baseline_fit.fit(frequency, spectrum).baseline
    
    # Create visualization
    plot_result(frequency, spectrum, baseline, lines, 
                save_path=save_path, detector_name=detector_name)
    
    return lines

def detect_from_fit(frequency: np.ndarray, spectrum: np.ndarray, fit_result, 
                   min_amplitude: float = GLOBAL_MIN_AMPLITUDE, 
                   baseline_level: float = GLOBAL_BASELINE_LEVEL,
                   min_prominence: float = GLOBAL_MIN_PROMINENCE,
                   save_path: Optional[str] = None,
                   detector_name: Optional[str] = None) -> List[Line]:
    """
    Convenience function to detect lines directly from baseline_fit Result object
    
    Extracts baseline from fit result and applies line detection algorithm.
    Ideal for integration with baseline_fit workflow.
    
    Args:
        frequency: Frequency array (Hz)
        spectrum: Original spectrum data
        fit_result: Result object from baseline_fit.fit()
        min_amplitude: Minimum peak amplitude threshold
        baseline_level: Baseline crossing threshold
        min_prominence: Minimum peak prominence
        save_path: Optional path to save plot
        detector_name: Optional detector identifier for labeling
        
    Returns:
        List of detected Line objects
    """
    
    return find_and_plot(frequency, spectrum, fit_result.baseline, 
                        min_amplitude, baseline_level, min_prominence, 
                        save_path, detector_name)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = ['Line', 'detect_lines', 'detect_lines_multiple', 
           'find_and_plot', 'detect_from_fit', 'plot_result']
