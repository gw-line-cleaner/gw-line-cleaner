#!/usr/bin/env python3
"""
Core line cleaning functions for gravitational wave spectral data.
Generalized to support arbitrary number of detectors.
"""

import logging
import numpy as np
from pathlib import Path
import csv
import os
from typing import Dict, List, Set
import tempfile
from io import StringIO
import contextlib

from . import baseline_fit
from . import simple_line_detector
from . import coherent_analyzer
from .coherent_analyzer import DetectorData, CoherentLineGroup

# Configure logging
LOGGER = logging.getLogger(__name__)


def _classify_width(width: float) -> str:
    """Match coherent_analyzer's width classification for consistency."""
    if width < coherent_analyzer.NARROW_LINE_THRESHOLD:
        return "narrow"
    if width > coherent_analyzer.WIDE_LINE_THRESHOLD:
        return "wide"
    return "medium"


def _write_csv(path: Path, header: list, rows_iter):
    """Utility to write CSV with header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows_iter:
            writer.writerow(row)


def clean_lines_from_PSD(
    freq: np.ndarray,
    detector_psds: Dict[str, np.ndarray],
    detector_clean: Dict[str, np.ndarray],
    out_dir: Path = None,
    out_name: str = "output",
    min_detectors: int = 2,
    export_csv: bool = True
) -> Dict[str, object]:
    """
    Detect single-interferometer spectral lines using coherent_analyzer and set cleaning arrays.
    Generalized to support arbitrary number of detectors.
    
    Lines that appear in fewer than min_detectors are considered instrument artifacts
    and will be cleaned. Lines appearing in >= min_detectors are considered potential
    astrophysical signals and are preserved.
    
    Args:
        freq: Frequency array (Hz), common to all detectors
        detector_psds: Dict mapping detector name to PSD array
                      e.g., {"H1": h1_psd, "L1": l1_psd, "V1": v1_psd}
        detector_clean: Dict mapping detector name to cleaning array (modified in place)
                       Values of 0 mean no cleaning; non-zero values are replacement PSD values
        out_dir: Output directory for CSV files (optional, set to None to skip file output)
        out_name: Base name for output files
        min_detectors: Minimum detectors for a line to be considered coherent (not cleaned)
        export_csv: Whether to export diagnostic CSV files
    
    Returns:
        Dict containing analysis results:
        - 'coherent_groups': List of CoherentLineGroup objects
        - 'detector_data': Dict of DetectorData per detector
        - 'detector_lines': Dict of detected lines per detector
        - 'coherent_freqs': Dict of coherent frequencies per detector
    
    Raises:
        ValueError: If fewer than 2 detectors provided or array shapes mismatch
    """
    detector_names = list(detector_psds.keys())
    
    if len(detector_names) < 2:
        raise ValueError("At least 2 detectors required for coherence analysis")
    
    # Validate array shapes
    n_points = len(freq)
    for det_name in detector_names:
        if len(detector_psds[det_name]) != n_points:
            raise ValueError(f"PSD array for {det_name} has wrong length: "
                           f"{len(detector_psds[det_name])} vs {n_points}")
        if len(detector_clean[det_name]) != n_points:
            raise ValueError(f"Clean array for {det_name} has wrong length")
    
    LOGGER.debug("clean_lines_from_PSD: freq shape=%s, detectors=%s", 
                 freq.shape, detector_names)

    try:
        # Initialize cleaning arrays to zero (no cleaning by default)
        for det_name in detector_names:
            detector_clean[det_name][:] = 0.0

        # Convert PSD to ASD for analysis modules (ASD = sqrt(PSD))
        detector_asds = {det: np.sqrt(psd) for det, psd in detector_psds.items()}

        LOGGER.debug("Converting PSD to ASD and starting coherent analysis")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save each detector's ASD data to temporary files
            temp_files = {}
            for det_name in detector_names:
                temp_file = os.path.join(temp_dir, f"{det_name}_temp.txt")
                with open(temp_file, 'w') as f:
                    for i in range(len(freq)):
                        f.write(f"{freq[i]:.8e} {detector_asds[det_name][i]:.8e}\n")
                temp_files[det_name] = temp_file

            LOGGER.debug("Running coherent analysis using coherent_analyzer module")

            # Use generalized coherent_analyzer
            with contextlib.redirect_stdout(StringIO()):
                coherent_groups, detector_data = coherent_analyzer.analyze_coherent_lines(
                    temp_files, min_detectors=min_detectors
                )

        # Extract per-detector data
        detector_lines: Dict[str, list] = {}
        detector_baselines_asd: Dict[str, np.ndarray] = {}
        detector_freqs: Dict[str, np.ndarray] = {}
        detector_spectra: Dict[str, np.ndarray] = {}
        
        for det_name in detector_names:
            det_data = detector_data[det_name]
            detector_lines[det_name] = det_data.lines
            detector_baselines_asd[det_name] = det_data.baseline
            detector_freqs[det_name] = det_data.freq
            detector_spectra[det_name] = det_data.spectrum

        LOGGER.debug("Coherent analysis results:")
        for det_name in detector_names:
            LOGGER.debug(f"  {det_name} total lines: {len(detector_lines[det_name])}")
        LOGGER.debug(f"  Coherent groups (>={min_detectors} detectors): {len(coherent_groups)}")

        # Convert ASD baselines to PSD baselines
        detector_baselines_psd = {det: baseline**2 for det, baseline in detector_baselines_asd.items()}

        # Build sets of coherent line frequencies per detector
        coherent_freqs: Dict[str, Set[float]] = {det: set() for det in detector_names}
        for group in coherent_groups:
            for det_name, line in group.detector_lines.items():
                coherent_freqs[det_name].add(line.frequency)

        # Export CSV files if requested
        if export_csv and out_dir is not None:
            out_dir = Path(out_dir)
            _export_analysis_results(
                out_dir, out_name, detector_names, detector_freqs, detector_spectra,
                detector_baselines_asd, detector_lines, coherent_freqs, coherent_groups, freq
            )

        # Apply cleaning masks
        for det_name in detector_names:
            det_lines = detector_lines[det_name]
            det_coherent_freqs = coherent_freqs[det_name]
            det_freq_grid = detector_freqs[det_name]
            det_baseline_psd = detector_baselines_psd[det_name]
            
            for line in det_lines:
                if line.frequency not in det_coherent_freqs:
                    # This line is NOT coherent - mark for cleaning
                    line_mask = (freq >= line.left_boundary) & (freq <= line.right_boundary)
                    idxs = np.where(line_mask)[0]
                    if len(idxs) > 0:
                        map_idx = np.searchsorted(det_freq_grid, freq[idxs], side='left')
                        map_idx = map_idx.clip(0, len(det_freq_grid)-1)
                        detector_clean[det_name][idxs] = det_baseline_psd[map_idx]

        # Export clean masks if requested
        if export_csv and out_dir is not None:
            for det_name in detector_names:
                _write_csv(
                    out_dir / f"clean_mask_{det_name}_{out_name}.csv",
                    ["freq_Hz", "clean_mask", "baseline_PSD_if_masked"],
                    (
                        (float(freq[i]), int(detector_clean[det_name][i] != 0.0),
                         float(detector_clean[det_name][i]) if detector_clean[det_name][i] != 0.0 else 0.0)
                        for i in range(len(freq))
                    )
                )

        # Log summary
        for det_name in detector_names:
            cleaning_points = int(np.sum(detector_clean[det_name] != 0.0))
            total_points = len(freq)
            LOGGER.debug(f"{det_name}: {cleaning_points}/{total_points} points marked for cleaning "
                        f"({100*cleaning_points/total_points:.2f}%)")

        return {
            'coherent_groups': coherent_groups,
            'detector_data': detector_data,
            'detector_lines': detector_lines,
            'coherent_freqs': coherent_freqs
        }

    except Exception as e:
        LOGGER.error(f"Error in clean_lines_from_PSD: {str(e)}")
        LOGGER.error("Setting cleaning arrays to unity (no cleaning applied downstream)")
        for det_name in detector_names:
            detector_clean[det_name][:] = 1.0
        raise


def _export_analysis_results(
    out_dir: Path, out_name: str, detector_names: List[str],
    detector_freqs: Dict, detector_spectra: Dict, detector_baselines_asd: Dict,
    detector_lines: Dict, coherent_freqs: Dict, coherent_groups: List, freq: np.ndarray
):
    """Export all analysis results to CSV files."""
    
    for det_name in detector_names:
        det_freq = detector_freqs[det_name]
        det_spectrum = detector_spectra[det_name]
        det_baseline_asd = detector_baselines_asd[det_name]
        det_lines = detector_lines[det_name]
        det_coherent_freqs = coherent_freqs[det_name]

        # Baseline table
        _write_csv(
            out_dir / f"baseline_{det_name}_{out_name}.csv",
            ["freq_Hz", "ASD", "baseline_ASD", "ratio_ASD_over_baseline"],
            (
                (float(det_freq[i]), float(det_spectrum[i]), float(det_baseline_asd[i]),
                 float(det_spectrum[i] / det_baseline_asd[i] if det_baseline_asd[i] > 0 else 1.0))
                for i in range(len(det_freq))
            )
        )

        # ASD table
        _write_csv(
            out_dir / f"ASD_{det_name}_{out_name}.csv",
            ["freq_Hz", "ASD"],
            ((float(det_freq[i]), float(det_spectrum[i])) for i in range(len(det_freq)))
        )

        # Lines table
        _write_csv(
            out_dir / f"lines_{det_name}_{out_name}.csv",
            ["detector", "frequency_Hz", "amplitude_ratio", "left_Hz", "right_Hz",
             "width_Hz", "width_class", "coherent", "action"],
            (
                (
                    det_name,
                    f"{ln.frequency:.8f}",
                    f"{ln.amplitude:.6f}",
                    f"{ln.left_boundary:.8f}",
                    f"{ln.right_boundary:.8f}",
                    f"{ln.width:.8f}",
                    _classify_width(ln.width),
                    "YES" if ln.frequency in det_coherent_freqs else "NO",
                    "preserved" if ln.frequency in det_coherent_freqs else "cleaned"
                )
                for ln in det_lines
            )
        )

    # Export coherent groups table
    _write_csv(
        out_dir / f"coherent_groups_{out_name}.csv",
        ["group_id", "avg_freq_Hz", "detector_count", "detectors", "freq_spread_Hz"],
        (
            (
                i + 1,
                f"{group.frequency:.8f}",
                group.detector_count,
                ",".join(sorted(group.detectors)),
                f"{max(l.frequency for l in group.detector_lines.values()) - min(l.frequency for l in group.detector_lines.values()):.8f}"
            )
            for i, group in enumerate(coherent_groups)
        )
    )

    # Summary table
    summary_rows = []
    for det_name in detector_names:
        lines = detector_lines[det_name]
        total = len(lines)
        narrow = sum(1 for ln in lines if _classify_width(ln.width) == "narrow")
        medium = sum(1 for ln in lines if _classify_width(ln.width) == "medium")
        wide = sum(1 for ln in lines if _classify_width(ln.width) == "wide")
        coherent = sum(1 for ln in lines if ln.frequency in coherent_freqs[det_name])
        summary_rows.append((det_name, total, narrow, medium, wide, coherent, total - coherent))
    
    _write_csv(
        out_dir / f"lines_summary_{out_name}.csv",
        ["detector", "total", "narrow", "medium", "wide", "coherent", "incoherent"],
        summary_rows
    )


def get_cleaning_mask(
    freq: np.ndarray,
    detector_psds: Dict[str, np.ndarray],
    min_detectors: int = 2
) -> Dict[str, np.ndarray]:
    """
    Convenience function to get cleaning masks without modifying input arrays.
    
    Args:
        freq: Frequency array (Hz)
        detector_psds: Dict mapping detector names to PSD arrays
        min_detectors: Minimum detectors for coherence
    
    Returns:
        Dict mapping detector names to boolean cleaning masks
        (True = should be cleaned, False = keep original)
    """
    detector_clean = {det: np.zeros_like(psd) for det, psd in detector_psds.items()}
    
    clean_lines_from_PSD(
        freq, detector_psds, detector_clean,
        out_dir=None, export_csv=False, min_detectors=min_detectors
    )
    
    # Convert to boolean masks
    return {det: (clean != 0.0) for det, clean in detector_clean.items()}


def apply_cleaning(
    freq: np.ndarray,
    detector_psds: Dict[str, np.ndarray],
    min_detectors: int = 2
) -> Dict[str, np.ndarray]:
    """
    Apply line cleaning and return cleaned PSD arrays.
    
    Args:
        freq: Frequency array (Hz)
        detector_psds: Dict mapping detector names to PSD arrays
        min_detectors: Minimum detectors for coherence
    
    Returns:
        Dict mapping detector names to cleaned PSD arrays
    """
    detector_clean = {det: np.zeros_like(psd) for det, psd in detector_psds.items()}
    
    clean_lines_from_PSD(
        freq, detector_psds, detector_clean,
        out_dir=None, export_csv=False, min_detectors=min_detectors
    )
    
    # Apply cleaning: where clean != 0, use clean value; otherwise keep original
    cleaned_psds = {}
    for det, psd in detector_psds.items():
        cleaned = psd.copy()
        mask = detector_clean[det] != 0.0
        cleaned[mask] = detector_clean[det][mask]
        cleaned_psds[det] = cleaned
    
    return cleaned_psds
