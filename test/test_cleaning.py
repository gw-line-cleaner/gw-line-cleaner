# SPDX-FileCopyrightText: 2026 Karl Wette
#
# SPDX-License-Identifier: MIT

"""Basic test suite."""

import numpy as np

from gw_line_cleaner import apply_cleaning


def test_no_lines():
    """Basic cleaning tests with no lines."""

    freq = np.linspace(100, 101, 1800)

    detector_PSDs = {
        "A1": 1e-45 * (1.0 + 0.5 * (freq - 101.5) ** 2),
        "B1": 1e-45 * (2.0 + 0.1 * (freq - 101.5)),
    }

    cleaned_PSDs, masks = apply_cleaning(freq, detector_PSDs, min_detectors=2)

    A_cleaned = masks["A1"].nonzero()[0]
    B_cleaned = masks["B1"].nonzero()[0]

    assert all(A_cleaned == [])
    assert all(B_cleaned == [])


def test_lines():
    """Basic cleaning tests with lines."""

    freq = np.linspace(100, 101, 1800)

    detector_PSDs = {
        "A1": 1e-45 * (1.0 + 0.5 * (freq - 101.5) ** 2),
        "B1": 1e-45 * (2.0 + 0.1 * (freq - 101.5)),
    }

    Aline = 100
    detector_PSDs["A1"][Aline] *= 20

    Bline = 300
    detector_PSDs["B1"][Bline] *= 20

    cleaned_PSDs, masks = apply_cleaning(freq, detector_PSDs, min_detectors=2)

    A_cleaned = masks["A1"].nonzero()[0]
    B_cleaned = masks["B1"].nonzero()[0]

    assert all(A_cleaned == [99, 100, 101])
    assert all(B_cleaned == [299, 300, 301])


def test_lines_signal():
    """Basic cleaning tests with lines and a coherent signal."""

    freq = np.linspace(100, 101, 1800)

    detector_PSDs = {
        "A1": 1e-45 * (1.0 + 0.5 * (freq - 100.5) ** 2),
        "B1": 1e-45 * (2.0 + 0.1 * (freq - 100.5)),
    }

    Aline = 100
    detector_PSDs["A1"][Aline] *= 20

    Bline = 300
    detector_PSDs["B1"][Bline] *= 20

    ABsignal = 1500
    detector_PSDs["A1"][ABsignal] *= 3
    detector_PSDs["B1"][ABsignal] = detector_PSDs["B1"][ABsignal]

    cleaned_PSDs, masks = apply_cleaning(freq, detector_PSDs, min_detectors=2)

    A_cleaned = masks["A1"].nonzero()[0]
    B_cleaned = masks["B1"].nonzero()[0]

    assert all(A_cleaned == [99, 100, 101])
    assert all(B_cleaned == [299, 300, 301])
