# SPDX-FileCopyrightText: 2026 Ye Zhou
#
# SPDX-License-Identifier: MIT

"""Multi-Detector Coherent Spectral Line Analyzer.

Generalized to support arbitrary number of detectors.
"""

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from . import baseline_fit, simple_line_detector
from .simple_line_detector import Line

logger = logging.getLogger(__name__)

# =============================================================================
# ADAPTIVE MATCHING PARAMETERS
# =============================================================================

# Parameters for Very Narrow Lines (Single-bin artifacts)
VERY_NARROW_LINE_THRESHOLD = 0.005
VERY_NARROW_FREQ_TOLERANCE_ABS = 1.0 / 3600.0
VERY_NARROW_AMPLITUDE_TOLERANCE = 0.90

# Standard Thresholds
NARROW_LINE_THRESHOLD = 2.0
WIDE_LINE_THRESHOLD = 10.0

# Narrow line matching
NARROW_FREQUENCY_TOLERANCE = 0.001
NARROW_AMPLITUDE_TOLERANCE = 0.80

# Wide line matching
WIDE_FREQUENCY_TOLERANCE = 0.02
WIDE_AMPLITUDE_TOLERANCE = 1.00
WIDE_MIN_OVERLAP_RATIO = 0.1

# Medium line matching
MEDIUM_FREQUENCY_TOLERANCE = 0.005
MEDIUM_AMPLITUDE_TOLERANCE = 1.0
MEDIUM_MIN_OVERLAP_RATIO = 0.05

# =============================================================================
# GENERALIZED DATA STRUCTURES
# =============================================================================


@dataclass
class DetectorData:
    """Container for single detector's analysis data."""

    name: str
    freq: np.ndarray
    spectrum: np.ndarray
    baseline: np.ndarray
    lines: List[Line]


@dataclass
class CoherentLinePair:
    """Represents a coherent line pair between two specific detectors."""

    frequency: float  # Average frequency
    detector1: str
    detector2: str
    line1: Line
    line2: Line
    amplitude_ratio: float
    frequency_difference: float


@dataclass
class CoherentLineGroup:
    """Represents a coherent line found across multiple detectors.

    Maps detector names to their respective Line objects.
    """

    frequency: float  # Average frequency across all detectors
    detector_lines: Dict[str, Line] = field(default_factory=dict)

    @property
    def detector_count(self) -> int:
        """Number of detectors."""
        return len(self.detector_lines)

    @property
    def detectors(self) -> List[str]:
        """List of detector names."""
        return list(self.detector_lines.keys())


# Legacy compatibility structure
@dataclass
class CoherentLine:
    """Legacy structure for backward compatibility with H1-L1 only code."""

    frequency: float
    h1_line: Line
    l1_line: Line
    amplitude_ratio: float
    frequency_difference: float


# =============================================================================
# MAIN ANALYSIS WORKFLOW - GENERALIZED
# =============================================================================


def analyze_coherent_lines(
    detector_asd_srcs: Dict[str, Union[str, np.ndarray]], min_detectors: int = 2
) -> Tuple[List[CoherentLineGroup], Dict[str, DetectorData]]:
    """Analyze coherent spectral lines across multiple detectors.

    Args:
        detector_asd_srcs (Dict[str, Union[str, np.ndarray]]):
            Dictionary mapping detector names to file paths containing ASDs:
                e.g., {"H1": "/path/to/h1.txt", "L1": "/path/to/l1.txt", "V1": "/path/to/v1.txt"}
            or mapping detector names to NumPy arrays containing ASDs:
                e.g., {"H1": np.array(...), "L1": np.array(...), "V1": np.array(...)}
        min_detectors (int): Minimum number of detectors a line must appear in to be considered coherent

    Returns:
        Tuple[List[CoherentLineGroup], Dict[str, DetectorData]]: Tuple of:
        - List of CoherentLineGroup objects (lines found in >= min_detectors)
        - Dictionary mapping detector names to DetectorData objects
    """
    if len(detector_asd_srcs) < 2:
        msg = "At least 2 detectors required for coherence analysis"
        raise ValueError(msg)

    # Process each detector
    detector_data: Dict[str, DetectorData] = {}

    for det_name, detector_asd_src in detector_asd_srcs.items():
        logger.info("Fitting baseline ASD of detector %s", det_name)
        freq, spectrum, result = baseline_fit.load_and_fit(detector_asd_src)
        assert result is not None
        logger.info("Detecting lines in %s spectrum", det_name)
        lines = simple_line_detector.detect_lines(
            freq, spectrum, result.baseline, detector_name=det_name
        )

        detector_data[det_name] = DetectorData(
            name=det_name,
            freq=freq,
            spectrum=spectrum,
            baseline=result.baseline,
            lines=lines,
        )

    # Find pairwise coherent lines
    logger.info("Finding pairwise coherent lines")
    all_pairs = _find_all_pairwise_coherent(detector_data)

    # Group into multi-detector coherent lines
    logger.info("Grouping multi-detector coherent lines")
    coherent_groups = _group_coherent_lines(all_pairs, detector_data, min_detectors)

    return coherent_groups, detector_data


def analyze_coherent_lines_from_data(
    detector_data: Dict[str, DetectorData], min_detectors: int = 2
) -> List[CoherentLineGroup]:
    """Analyze coherent lines from pre-loaded detector data.

    Useful when data is already in memory.
    """
    if len(detector_data) < 2:
        msg = "At least 2 detectors required for coherence analysis"
        raise ValueError(msg)

    all_pairs = _find_all_pairwise_coherent(detector_data)
    coherent_groups = _group_coherent_lines(all_pairs, detector_data, min_detectors)

    return coherent_groups


# =============================================================================
# BACKWARD COMPATIBILITY - LEGACY API
# =============================================================================


def analyze_coherent_lines_legacy(
    h1_file: str, l1_file: str
) -> Tuple[List[CoherentLine], dict]:
    """Legacy interface for backward compatibility.

    Wraps the new generalized interface to match the old API.

    Returns data in the old format with 'h1_*' and 'l1_*' keys.
    """
    detector_files = {"H1": h1_file, "L1": l1_file}
    coherent_groups, detector_data = analyze_coherent_lines(
        detector_files, min_detectors=2
    )

    # Convert to legacy CoherentLine format
    legacy_coherent = []
    for group in coherent_groups:
        if "H1" in group.detector_lines and "L1" in group.detector_lines:
            h1_line = group.detector_lines["H1"]
            l1_line = group.detector_lines["L1"]

            # Line.amplitude is already the dimensionless peak-to-baseline ASD ratio.
            h1_enhancement = h1_line.amplitude
            l1_enhancement = l1_line.amplitude
            amp_ratio = abs(h1_enhancement - l1_enhancement) / max(
                h1_enhancement, l1_enhancement
            )

            legacy_coherent.append(
                CoherentLine(
                    frequency=group.frequency,
                    h1_line=h1_line,
                    l1_line=l1_line,
                    amplitude_ratio=amp_ratio,
                    frequency_difference=abs(h1_line.frequency - l1_line.frequency),
                )
            )

    # Convert to legacy data format
    h1_data = detector_data["H1"]
    l1_data = detector_data["L1"]

    legacy_data = {
        "h1_freq": h1_data.freq,
        "h1_spectrum": h1_data.spectrum,
        "h1_baseline": h1_data.baseline,
        "h1_lines": h1_data.lines,
        "l1_freq": l1_data.freq,
        "l1_spectrum": l1_data.spectrum,
        "l1_baseline": l1_data.baseline,
        "l1_lines": l1_data.lines,
    }

    return legacy_coherent, legacy_data


# =============================================================================
# PAIRWISE COHERENCE DETECTION
# =============================================================================


def _find_all_pairwise_coherent(
    detector_data: Dict[str, DetectorData],
) -> List[CoherentLinePair]:
    """Find all pairwise coherent lines between all detector combinations."""
    all_pairs = []
    detector_names = list(detector_data.keys())

    # Iterate over all unique pairs of detectors
    for det1_name, det2_name in itertools.combinations(detector_names, 2):
        det1 = detector_data[det1_name]
        det2 = detector_data[det2_name]

        pairs = _find_coherent_lines_between_detectors(
            det1_name,
            det1.lines,
            det1.freq,
            det1.baseline,
            det2_name,
            det2.lines,
            det2.freq,
            det2.baseline,
        )
        all_pairs.extend(pairs)

    return all_pairs


def _find_coherent_lines_between_detectors(
    det1_name: str,
    det1_lines: List[Line],
    det1_freq: np.ndarray,
    det1_baseline: np.ndarray,
    det2_name: str,
    det2_lines: List[Line],
    det2_freq: np.ndarray,
    det2_baseline: np.ndarray,
) -> List[CoherentLinePair]:
    """Find coherent lines between two specific detectors."""
    coherent_pairs = []
    used_det2_indices: Set[int] = set()

    for det1_line in det1_lines:
        # Line.amplitude is already the dimensionless peak-to-baseline ASD ratio.
        det1_enhancement = det1_line.amplitude

        candidates = []

        for i, det2_line in enumerate(det2_lines):
            if i in used_det2_indices:
                continue

            det2_enhancement = det2_line.amplitude

            params = _get_adaptive_parameters(det1_line, det2_line)

            # Frequency check
            freq_diff = abs(det1_line.frequency - det2_line.frequency)

            if params.get("use_abs_freq", False):
                freq_metric = freq_diff
            else:
                freq_metric = freq_diff / min(det1_line.frequency, det2_line.frequency)

            if freq_metric > params["freq_tol"]:
                continue

            # High Precision Frequency Bypass
            is_precise_match = freq_diff < 0.1

            # Overlap check
            overlap_ratio = _calculate_overlap(det1_line, det2_line)
            if (
                params["use_overlap"]
                and overlap_ratio < params["overlap_min"]
                and not is_precise_match
            ):
                continue

            # Amplitude check
            enh_ratio_diff = abs(det1_enhancement - det2_enhancement) / max(
                det1_enhancement, det2_enhancement
            )
            if enh_ratio_diff > params["amp_tol"] and not is_precise_match:
                continue

            # Scoring
            relative_freq_diff = freq_diff / min(
                det1_line.frequency, det2_line.frequency
            )

            if params["type"] == "very_narrow":
                norm_freq_score = freq_diff / VERY_NARROW_FREQ_TOLERANCE_ABS
                score = norm_freq_score * 3.0 + enh_ratio_diff * 0.5
            elif params["type"] == "narrow":
                score = relative_freq_diff * 3.0 + enh_ratio_diff * 0.5
            elif params["type"] == "wide":
                score = (
                    relative_freq_diff * 1.0
                    + enh_ratio_diff * 0.5
                    + (1.0 - overlap_ratio) * 2.0
                )
            else:
                score = (
                    relative_freq_diff * 2.0
                    + enh_ratio_diff * 0.5
                    + (1.0 - overlap_ratio) * 1.0
                )

            candidates.append((det2_line, i, score, det2_enhancement, params["type"]))

        if candidates:
            candidates.sort(key=lambda x: x[2])
            best_det2_line, best_det2_idx, _, l2_enh, _ = candidates[0]

            amp_ratio = abs(det1_enhancement - l2_enh) / max(det1_enhancement, l2_enh)

            coherent_pairs.append(
                CoherentLinePair(
                    frequency=(det1_line.frequency + best_det2_line.frequency) / 2,
                    detector1=det1_name,
                    detector2=det2_name,
                    line1=det1_line,
                    line2=best_det2_line,
                    amplitude_ratio=amp_ratio,
                    frequency_difference=abs(
                        det1_line.frequency - best_det2_line.frequency
                    ),
                )
            )

            used_det2_indices.add(best_det2_idx)

    return coherent_pairs


def _group_coherent_lines(
    pairs: List[CoherentLinePair],
    detector_data: Dict[str, DetectorData],
    min_detectors: int,
) -> List[CoherentLineGroup]:
    """Group pairwise coherent lines into multi-detector groups.

    Uses frequency-based clustering to identify the same physical line
    across multiple detector pairs.
    """
    if not pairs:
        return []

    # Sort pairs by frequency
    sorted_pairs = sorted(pairs, key=lambda p: p.frequency)

    groups: List[CoherentLineGroup] = []

    for pair in sorted_pairs:
        # Try to find an existing group this pair belongs to
        merged = False
        for group in groups:
            # Check if this pair's frequency is close to the group's frequency
            freq_diff = abs(pair.frequency - group.frequency)
            relative_diff = (
                freq_diff / group.frequency if group.frequency > 0 else float("inf")
            )

            if relative_diff < 0.01:  # 1% frequency tolerance for grouping
                # Add both detectors to the group
                if pair.detector1 not in group.detector_lines:
                    group.detector_lines[pair.detector1] = pair.line1
                if pair.detector2 not in group.detector_lines:
                    group.detector_lines[pair.detector2] = pair.line2

                # Update group frequency to average
                all_freqs = [line.frequency for line in group.detector_lines.values()]
                group.frequency = np.mean(all_freqs)
                merged = True
                break

        if not merged:
            # Create new group
            new_group = CoherentLineGroup(
                frequency=pair.frequency,
                detector_lines={pair.detector1: pair.line1, pair.detector2: pair.line2},
            )
            groups.append(new_group)

    # Filter by minimum detector count
    filtered_groups = [g for g in groups if g.detector_count >= min_detectors]

    return sorted(filtered_groups, key=lambda g: g.frequency)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _classify_line_width(line: Line) -> str:
    """Classify line by width category."""
    if line.width < VERY_NARROW_LINE_THRESHOLD:
        return "very_narrow"
    elif line.width < NARROW_LINE_THRESHOLD:
        return "narrow"
    elif line.width > WIDE_LINE_THRESHOLD:
        return "wide"
    else:
        return "medium"


def _get_adaptive_parameters(line1: Line, line2: Line) -> dict:
    """Get matching parameters based on line width classification."""
    type1 = _classify_line_width(line1)
    type2 = _classify_line_width(line2)

    # Case 1: Very Narrow (Glitch-like)
    if type1 == "very_narrow" or type2 == "very_narrow":
        return {
            "type": "very_narrow",
            "use_abs_freq": True,
            "freq_tol": VERY_NARROW_FREQ_TOLERANCE_ABS,
            "amp_tol": VERY_NARROW_AMPLITUDE_TOLERANCE,
            "overlap_min": 0.0,
            "use_overlap": False,
        }

    # Case 2: Both Narrow
    elif type1 == "narrow" and type2 == "narrow":
        return {
            "type": "narrow",
            "use_abs_freq": False,
            "freq_tol": NARROW_FREQUENCY_TOLERANCE,
            "amp_tol": NARROW_AMPLITUDE_TOLERANCE,
            "overlap_min": 0.0,
            "use_overlap": False,
        }

    # Case 3: Both Wide
    elif type1 == "wide" and type2 == "wide":
        return {
            "type": "wide",
            "use_abs_freq": False,
            "freq_tol": WIDE_FREQUENCY_TOLERANCE,
            "amp_tol": WIDE_AMPLITUDE_TOLERANCE,
            "overlap_min": WIDE_MIN_OVERLAP_RATIO,
            "use_overlap": True,
        }

    # Case 4: Medium / Mixed
    else:
        return {
            "type": "medium",
            "use_abs_freq": False,
            "freq_tol": MEDIUM_FREQUENCY_TOLERANCE,
            "amp_tol": MEDIUM_AMPLITUDE_TOLERANCE,
            "overlap_min": MEDIUM_MIN_OVERLAP_RATIO,
            "use_overlap": True,
        }


def _calculate_overlap(line1: Line, line2: Line) -> float:
    """Calculate frequency overlap ratio between two spectral lines."""
    overlap_start = max(line1.left_boundary, line2.left_boundary)
    overlap_end = min(line1.right_boundary, line2.right_boundary)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_width = overlap_end - overlap_start
    total_width = max(line1.right_boundary, line2.right_boundary) - min(
        line1.left_boundary, line2.left_boundary
    )
    return overlap_width / total_width if total_width > 0 else 0.0


# =============================================================================
# VISUALIZATION - GENERALIZED
# =============================================================================


def plot_results(
    coherent_groups: List[CoherentLineGroup],
    detector_data: Dict[str, DetectorData],
    save_path: Optional[str] = None,
) -> None:
    """Plot results for arbitrary number of detectors.

    Creates one subplot per detector plus one for coherent lines overlay.
    """
    n_detectors = len(detector_data)
    n_plots = n_detectors + 1  # One per detector + one overlay

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # Plot each detector
    for idx, (det_name, det_data) in enumerate(sorted(detector_data.items())):
        _plot_single_detector(
            axes[idx], det_data, f"{det_name} Detected Spectral Lines"
        )

    # Plot coherent overlay
    _plot_coherent_overlay(axes[-1], coherent_groups, detector_data)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def _plot_single_detector(ax, det_data: DetectorData, title: str) -> None:
    """Plot single detector spectrum with detected lines."""
    freq = det_data.freq
    spectrum = det_data.spectrum
    baseline = det_data.baseline
    lines = det_data.lines

    ax.loglog(
        freq,
        spectrum,
        "k-",
        alpha=0.7,
        linewidth=0.3,
        label=f"{det_data.name} Spectrum",
    )
    ax.loglog(freq, baseline, "r-", linewidth=1.5, label="Baseline")

    if lines:
        line_freqs = [line.frequency for line in lines]
        line_amps = [spectrum[np.argmin(np.abs(freq - lf))] for lf in line_freqs]
        ax.loglog(
            line_freqs,
            line_amps,
            "^",
            color="orange",
            markersize=6,
            label=f"Lines ({len(lines)})",
        )

        for line in lines:
            line_type = _classify_line_width(line)
            color = {
                "very_narrow": "purple",
                "narrow": "lightblue",
                "wide": "lightcoral",
            }.get(line_type, "orange")
            ax.axvspan(line.left_boundary, line.right_boundary, alpha=0.15, color=color)

    ax.set_ylim(np.min(spectrum) * 0.5, np.max(spectrum) * 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ASD (1/√Hz)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def _plot_coherent_overlay(
    ax, coherent_groups: List[CoherentLineGroup], detector_data: Dict[str, DetectorData]
) -> None:
    """Plot overlay of all detectors with coherent lines highlighted."""
    colors = plt.cm.tab10.colors

    for idx, (det_name, det_data) in enumerate(sorted(detector_data.items())):
        color = colors[idx % len(colors)]
        ax.loglog(
            det_data.freq,
            det_data.spectrum,
            "-",
            color=color,
            alpha=0.4,
            linewidth=0.3,
            label=det_name,
        )
        ax.loglog(
            det_data.freq,
            det_data.baseline,
            "--",
            color=color,
            alpha=0.8,
            linewidth=1.5,
            label=f"{det_name} Baseline",
        )

    if coherent_groups:
        for group in coherent_groups:
            # Find extent of coherent line across all detectors
            all_lefts = [line.left_boundary for line in group.detector_lines.values()]
            all_rights = [line.right_boundary for line in group.detector_lines.values()]
            lb = min(all_lefts)
            rb = max(all_rights)
            ax.axvspan(lb, rb, alpha=0.15, color="red")

            # Mark peaks
            for det_name, line in group.detector_lines.items():
                det_data = detector_data[det_name]
                idx = np.argmin(np.abs(det_data.freq - line.frequency))
                amp = det_data.spectrum[idx]
                ax.loglog(
                    [line.frequency], [amp], "o", color="red", markersize=4, alpha=0.8
                )

    # Get y limits from first detector
    first_det = list(detector_data.values())[0]
    ax.set_ylim(np.min(first_det.spectrum) * 0.5, np.max(first_det.spectrum) * 2)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ASD (1/√Hz)")
    ax.set_title(
        f"Multi-Detector Coherent Lines ({len(coherent_groups)} found)",
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


# =============================================================================
# SUMMARY FUNCTIONS - GENERALIZED
# =============================================================================


def print_summary(
    coherent_groups: List[CoherentLineGroup], detector_data: Dict[str, DetectorData]
) -> None:
    """Print summary of coherent line analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-DETECTOR COHERENT SPECTRAL LINES ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Detectors analyzed: {', '.join(sorted(detector_data.keys()))}")
    logger.info(f"Total coherent line groups: {len(coherent_groups)}")

    if not coherent_groups:
        logger.info("No coherent lines detected")
        return

    logger.info("\n" + "-" * 80)
    logger.info(
        f"{'No.':<4} {'Freq(Hz)':<12} {'Detectors':<20} {'Freq Spread(Hz)':<15}"
    )
    logger.info("-" * 80)

    for i, group in enumerate(coherent_groups, 1):
        det_list = ",".join(sorted(group.detectors))
        freqs = [line.frequency for line in group.detector_lines.values()]
        freq_spread = max(freqs) - min(freqs) if len(freqs) > 1 else 0
        logger.info(
            f"{i:<4} {group.frequency:<12.2f} {det_list:<20} {freq_spread:<15.4f}"
        )


def print_all_lines_detailed(
    coherent_groups: List[CoherentLineGroup], detector_data: Dict[str, DetectorData]
) -> None:
    """Print detailed information about all detected lines."""
    # Build coherent frequency sets per detector
    coherent_freqs: Dict[str, Set[float]] = {det: set() for det in detector_data.keys()}
    for group in coherent_groups:
        for det_name, line in group.detector_lines.items():
            coherent_freqs[det_name].add(line.frequency)

    logger.info("\n" + "=" * 100)
    logger.info("DETAILED SPECTRAL LINE ANALYSIS - ALL DETECTED LINES")
    logger.info("=" * 100)

    for det_name in sorted(detector_data.keys()):
        det_data = detector_data[det_name]
        lines = det_data.lines
        det_coherent = coherent_freqs[det_name]

        logger.info(f"\n{det_name} DETECTOR LINES (Total: {len(lines)})")
        logger.info("-" * 100)
        logger.info(
            f"{'No.':<4} {'Frequency':<12} {'Amplitude':<10} {'Width':<10} {'Type':<12} {'Coherent':<10}"
        )
        logger.info("-" * 100)

        for i, line in enumerate(lines, 1):
            line_type = _classify_line_width(line)
            is_coherent = line.frequency in det_coherent
            coherent_flag = "YES" if is_coherent else "NO"
            logger.info(
                f"{i:<4} {line.frequency:<12.2f} {line.amplitude:<10.2f} {line.width:<10.4f} {line_type.upper():<12} {coherent_flag:<10}"
            )


def print_detector_statistics(
    detector_data: Dict[str, DetectorData], coherent_groups: List[CoherentLineGroup]
) -> None:
    """Print per-detector statistics."""
    logger.info("\n" + "=" * 80)
    logger.info("PER-DETECTOR STATISTICS")
    logger.info("=" * 80)

    for det_name in sorted(detector_data.keys()):
        det = detector_data[det_name]
        lines = det.lines
        total = len(lines)

        # Count by width class
        vn = sum(1 for line in lines if _classify_line_width(line) == "very_narrow")
        n = sum(1 for line in lines if _classify_line_width(line) == "narrow")
        m = sum(1 for line in lines if _classify_line_width(line) == "medium")
        w = sum(1 for line in lines if _classify_line_width(line) == "wide")

        # Count coherent
        coherent_count = sum(
            1 for group in coherent_groups if det_name in group.detector_lines
        )

        logger.info(f"\n{det_name}: {total} total lines")
        if total > 0:
            logger.info(f"  Very Narrow: {vn} ({100*vn/total:.1f}%)")
            logger.info(f"  Narrow: {n} ({100*n/total:.1f}%)")
            logger.info(f"  Medium: {m} ({100*m/total:.1f}%)")
            logger.info(f"  Wide: {w} ({100*w/total:.1f}%)")
        logger.info(f"  In coherent groups: {coherent_count}")


def analyze_and_plot(
    detector_files: Dict[str, str],
    save_path: Optional[str] = None,
    min_detectors: int = 2,
) -> Tuple[List[CoherentLineGroup], Dict[str, DetectorData]]:
    """Complete analysis workflow: analyze and plot results.

    Args:
        detector_files (Dict[str, str]): Dictionary mapping detector names to file paths
        save_path (Optional[str]): Optional path to save plot
        min_detectors (int): Minimum detectors for coherence

    Returns:
        Tuple[List[CoherentLineGroup], Dict[str, DetectorData]]:
            Tuple of (coherent_groups, detector_data)
    """
    coherent_groups, detector_data = analyze_coherent_lines(
        detector_files, min_detectors
    )
    print_all_lines_detailed(coherent_groups, detector_data)
    print_detector_statistics(detector_data, coherent_groups)
    print_summary(coherent_groups, detector_data)
    plot_results(coherent_groups, detector_data, save_path)
    return coherent_groups, detector_data


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "DetectorData",
    "CoherentLinePair",
    "CoherentLineGroup",
    "CoherentLine",  # Legacy
    "analyze_coherent_lines",
    "analyze_coherent_lines_from_data",
    "analyze_coherent_lines_legacy",  # Backward compatibility
    "analyze_and_plot",
    "plot_results",
    "print_summary",
    "print_all_lines_detailed",
    "print_detector_statistics",
    "VERY_NARROW_LINE_THRESHOLD",
    "NARROW_LINE_THRESHOLD",
    "WIDE_LINE_THRESHOLD",
]
