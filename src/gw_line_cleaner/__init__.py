"""
GW Line Cleaner - Remove line artifacts from gravitational wave data

A Python package for detecting and removing spectral line artifacts
from gravitational wave detector data. Supports analysis across
arbitrary numbers of detectors (LIGO, Virgo, KAGRA, etc.).
"""

__version__ = "0.1.0"

# Import main classes and functions for public API
from .baseline_fit import (
    load,
    fit,
    load_and_fit,
    load_and_fit_multiple,
    batch,
    Result,
)

from .simple_line_detector import (
    Line,
    detect_lines,
    detect_lines_multiple,
    find_and_plot,
    detect_from_fit,
    plot_result,
)

from .coherent_analyzer import (
    DetectorData,
    CoherentLinePair,
    CoherentLineGroup,
    CoherentLine,
    analyze_coherent_lines,
    analyze_coherent_lines_from_data,
    analyze_coherent_lines_legacy,
    analyze_and_plot,
    plot_results,
    print_summary,
    print_all_lines_detailed,
    print_detector_statistics,
    VERY_NARROW_LINE_THRESHOLD,
    NARROW_LINE_THRESHOLD,
    WIDE_LINE_THRESHOLD,
)

from .cleaner import (
    clean_lines_from_PSD,
    get_cleaning_mask,
    apply_cleaning,
)

__all__ = [
    # Version
    "__version__",
    # Baseline fitting
    "load",
    "fit",
    "load_and_fit",
    "load_and_fit_multiple",
    "batch",
    "Result",
    # Line detection
    "Line",
    "detect_lines",
    "detect_lines_multiple",
    "find_and_plot",
    "detect_from_fit",
    "plot_result",
    # Coherent analysis
    "DetectorData",
    "CoherentLinePair",
    "CoherentLineGroup",
    "CoherentLine",
    "analyze_coherent_lines",
    "analyze_coherent_lines_from_data",
    "analyze_coherent_lines_legacy",
    "analyze_and_plot",
    "plot_results",
    "print_summary",
    "print_all_lines_detailed",
    "print_detector_statistics",
    "VERY_NARROW_LINE_THRESHOLD",
    "NARROW_LINE_THRESHOLD",
    "WIDE_LINE_THRESHOLD",
    # Cleaning
    "clean_lines_from_PSD",
    "get_cleaning_mask",
    "apply_cleaning",
]
