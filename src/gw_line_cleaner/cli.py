# SPDX-FileCopyrightText: 2026 Karl Wette
#
# SPDX-License-Identifier: MIT

"""Command-line parser."""

import argparse
import logging
import os
import sys
import tempfile
import textwrap
from contextlib import contextmanager
from pathlib import Path

import h5py
import lal
import lalpulsar as lalp
import matplotlib
import numpy as np
from coloredlogs import ColoredFormatter as _Formatter

from . import __version__, apply_cleaning

__all__ = ["cli"]

# Configure logging
logger = logging.getLogger()
if not logger.hasHandlers():
    _log_handler = logging.StreamHandler(sys.stdout)
    _log_handler.setFormatter(
        _Formatter(
            fmt="[%(asctime)s] %(levelname)+8s: %(message)s",
        )
    )
    logger.addHandler(_log_handler)


@contextmanager
def silence_lal_errors():
    """Silence errors from LAL."""
    save_debug_level = lal.GetDebugLevel()
    try:
        lal.ClobberDebugLevel(
            save_debug_level
            & ~(lal.LALERRORBIT | lal.LALWARNINGBIT | lal.LALINFOBIT | lal.LALTRACEBIT)
        )
        yield
    finally:
        lal.ClobberDebugLevel(save_debug_level)


def absolute_path(arg):
    """Return fully-resolved path."""
    return Path(arg).resolve()


def params_spec(arg):
    """Return a continuous wave signal injection parameters file and section name."""
    if ":" in arg:
        path, section = arg.split(":", maxsplit=1)
    else:
        path = arg
        section = None
    return absolute_path(path), section


def parse_command_line(argv):
    """Parse command line."""

    # Create command-line parser
    parser = argparse.ArgumentParser(
        description="Remove line artifacts from gravitational wave data"
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: %(default)s).",
    )
    parser.add_argument(
        "-t",
        "--temp-dir",
        type=absolute_path,
        default=None,
        help="Temporary directory to write intermediate files.",
    )
    parser.add_argument(
        "-i",
        "--input-SFTs",
        type=absolute_path,
        required=True,
        help="""
        Input SFTs. Either a directory containing SFT files,
        or a text file containing SFT filenames.
        """,
    )
    parser.add_argument(
        "-o",
        "--output-SFT-dir",
        type=absolute_path,
        required=True,
        help="Directory in which to write output SFTs.",
    )
    parser.add_argument(
        "-I",
        "--inject-params",
        type=params_spec,
        help="""
        If given, configuration file with parameters of a continuous wave signal
        to be injected into the input SFTs before cleaning.
        If argument contains a colon, the string before the colon is the filename,
        and the string after the colon is the section to read the parameters from.
        """,
    )
    parser.add_argument(
        "-D",
        "--inject-as-depth",
        action="store_true",
        help="""
        If true, injection amplitudes are interpreted as (inverse) sensitivity
        depths, i.e. (h0, aPlus, aCross) => (h0, aPlus, aCross) * sqrt(Sh).
        """,
    )
    parser.add_argument(
        "-B",
        "--Fstat-before-params",
        type=params_spec,
        help="""
        If given, configuration file with parameters of a continuous wave signal
        at which to compute the F-statistic before cleaning.
        If argument contains a colon, the string before the colon is the filename,
        and the string after the colon is the section to read the parameters from.
        """,
    )
    parser.add_argument(
        "-A",
        "--Fstat-after-params",
        type=params_spec,
        help="""
        If given, configuration file with parameters of a continuous wave signal
        at which to compute the F-statistic after cleaning.
        If argument contains a colon, the string before the colon is the filename,
        and the string after the colon is the section to read the parameters from.
        """,
    )
    parser.add_argument(
        "-O",
        "--output-stats",
        type=absolute_path,
        help="""
        Path to HDF5 file where various statistics will be written.
        """,
    )
    parser.add_argument(
        "-S",
        "--output-spectra",
        action="store_true",
        help="""
        Include amplitude spectral densities before/after cleaning in output
        statistics.
        """,
    )
    parser.add_argument(
        "-P",
        "--output-spectra-plots",
        type=absolute_path,
        help="""
        Path to image file where plots of amplitude spectral densities before/after
        cleaning will be written.
        """,
    )
    parser.add_argument(
        "-e",
        "--earth-ephemeris",
        type=str,
        default="earth00-40-DE405.dat.gz",
        help="Earth ephemeris file to use.",
    )
    parser.add_argument(
        "-s",
        "--sun-ephemeris",
        type=str,
        default="sun00-40-DE405.dat.gz",
        help="Sun ephemeris file to use.",
    )

    return parser.parse_args(argv)


def load_params(params_spec, amp=True):
    """Load CW signal params."""
    path, section = params_spec
    data = lal.ParseDataFile(str(path))
    params = lalp.PulsarParams()
    lalp.ReadPulsarParams(params, data, section, None)
    params_dict = {}
    if amp:
        for fn in ("psi", "phi0", "aPlus", "aCross"):
            params_dict[fn] = getattr(params.Amp, fn)
    for fn in (
        "refTime",
        "Alpha",
        "Delta",
        "fkdot",
        "asini",
        "period",
        "ecc",
        "tp",
        "argp",
    ):
        if fn == "fkdot":
            params_dict[fn] = [float(f) for f in params.Doppler.fkdot]
        else:
            params_dict[fn] = float(getattr(params.Doppler, fn))

    return params, params_dict


def parse_SFT_filename(SFT_filename):
    """Parse SFT filename into specification structure."""
    SFT_fn_spec = lalp.SFTFilenameSpec()
    while True:
        try:
            with silence_lal_errors():
                lalp.ParseSFTFilenameIntoSpec(SFT_fn_spec, SFT_filename)
            return SFT_fn_spec
        except RuntimeError:

            # Handle pre-SFTv3 filenames with extra underscores
            if "_" in SFT_filename:
                SFT_fn_parts = SFT_filename.rsplit("_", 1)
                SFT_filename = "".join(SFT_fn_parts)
                continue

            # Something else went wrong
            raise


def compute_F_statistic(msg, SFT_filenames, Fstat_params, edat):
    """Compute the F-statistic on the given SFTs."""

    # Get search params
    Fstat_params, Fstat_params_dict = load_params(Fstat_params, amp=False)
    doppler = Fstat_params.Doppler

    # Create input data
    logger.info("Creating input for F-statistic")
    SFT_catalog = lalp.SFTdataFind(";".join(SFT_filenames), None)
    par_spin_range = lalp.PulsarSpinRange()
    lalp.InitPulsarSpinRangeFromSpins(
        par_spin_range, doppler.refTime, doppler.fkdot, doppler.fkdot
    )
    f_min, f_max = lalp.CWSignalCoveringBand(
        SFT_catalog.data[0].header.epoch,
        SFT_catalog.data[-1].header.epoch,
        par_spin_range,
        doppler.asini,
        doppler.period,
        doppler.ecc,
    )
    Fstat_input_args = lalp.FstatOptionalArgs(lalp.FstatOptionalArgsDefaults)
    Fstat_input_args.FstatMethod = lalp.FMETHOD_DEMOD_BEST
    Fstat_input = lalp.CreateFstatInput(
        SFT_catalog,
        f_min,
        f_max,
        SFT_catalog.data[0].header.deltaF,
        edat,
        Fstat_input_args,
    )

    # Compute F-statistic
    logger.info("Computing F-statistic %s...", msg)
    Fstat_res = 0
    Fstat_res = lalp.ComputeFstat(
        Fstat_res, Fstat_input, doppler, 1, lalp.FSTATQ_2F + lalp.FSTATQ_2F_PER_DET
    )
    logger.info("Computing F-statistic %s... done", msg)
    Fstat_params_dict["twoF"] = float(Fstat_res.twoF[0])
    for X in range(Fstat_res.numDetectors):
        detector_name = "".join(chr(c) for c in Fstat_res.detectorNames[X] if c > 0)
        Fstat_params_dict["twoF_" + detector_name] = float(Fstat_res.twoFPerDet(X)[0])
    logger.info("Computed F-statistic parameters: %s", Fstat_params_dict)

    return Fstat_params_dict


def write_out_stats_to_h5(h5_group, d):
    """Write outputs stats recursively to HDF5 file."""
    for k, v in d.items():
        if isinstance(v, dict):
            h5_sub_group = h5_group.create_group(k)
            write_out_stats_to_h5(h5_sub_group, v)
        else:
            if isinstance(v, np.ndarray):
                h5_group.create_dataset(
                    k, data=v, compression="gzip", compression_opts=6
                )
            else:
                h5_group.create_dataset(k, data=v)


def cli(*argv):
    """Command-line parser entry point."""

    # Parse command line
    args = parse_command_line(argv if argv else sys.argv[1:])

    # Set log level
    logger.setLevel(args.log_level)

    # Parse input SFT argument
    if args.input_SFTs.is_file():
        input_SFT_filenames = []
        with args.input_SFTs.open() as f:
            for line in f:
                line = line.strip()
                if line != "" and not line.startswith("#"):
                    input_SFT_filenames.append(Path(line))
        logger.info(
            "Found %i SFT filenames in file %s",
            len(input_SFT_filenames),
            args.input_SFTs,
        )
    else:
        input_SFT_filenames = list(args.input_SFTs.rglob("*.sft"))
        logger.info(
            "Found %i SFTs in directory %s", len(input_SFT_filenames), args.input_SFTs
        )
    input_SFT_filenames = sorted([str(p.resolve()) for p in input_SFT_filenames])
    for input_SFT_filename in input_SFT_filenames:
        logger.info("Input SFT filename: %s", input_SFT_filename)

    # Load ephemerides if needed
    if args.inject_params or args.Fstat_before_params or args.Fstat_after_params:
        edat = lalp.InitBarycenter(args.earth_ephemeris, args.sun_ephemeris)
    else:
        edat = None

    # Output statistics
    out_stats = {"version": __version__}
    out_spectra = {}

    # Create temporary directory
    if args.temp_dir is not None:
        args.temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="gw-line-cleaner-", dir=args.temp_dir
    ) as temp_dir:
        os.chdir(temp_dir)

        # Make output SFT directory
        args.output_SFT_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", args.output_SFT_dir)

        # Inject signal
        if args.inject_params:
            logger.info(
                "Injecting signal from file %s, section %s", *args.inject_params
            )
            out_stats["inj_intersect_cleaned"] = {}

            # Load injection parameters
            inj_params, out_stats["inject_params"] = load_params(args.inject_params)

            if args.inject_as_depth:

                # Load all SFTs
                SFT_catalog = lalp.SFTdataFind(";".join(input_SFT_filenames), None)
                SFTs = lalp.LoadMultiSFTs(
                    SFT_catalog,
                    inj_params.Doppler.fkdot[0] - 0.5,
                    inj_params.Doppler.fkdot[0] + 0.5,
                )

                # Compute PSD
                PSD = lalp.ComputePSDfromSFTs(
                    inputSFTs=SFTs,
                    blocksRngMed=lalp.FstatOptionalArgsDefaults.runningMedianWindow,
                    PSDmthopSFTs=lalp.MATH_OP_HARMONIC_SUM,
                    PSDmthopIFOs=lalp.MATH_OP_HARMONIC_SUM,
                    normalizeByTotalNumSFTs=True,
                    FreqMin=inj_params.Doppler.fkdot[0],
                    FreqBand=0,
                )
                sqrt_Sh = float(np.sqrt(PSD.data[0]))

                # Rescale aPlus and aCross
                inj_params.Amp.aPlus *= sqrt_Sh
                inj_params.Amp.aCross *= sqrt_Sh
                out_stats["inject_params"]["aPlus"] *= sqrt_Sh
                out_stats["inject_params"]["aCross"] *= sqrt_Sh

            logger.info("Injection parameters: %s", out_stats["inject_params"])

            # Make directory for injection SFTs
            inj_SFT_dir = Path(temp_dir) / "injected_SFTs"
            inj_SFT_dir.mkdir(parents=True)
            logger.info("Injected SFTs directory: %s", inj_SFT_dir)

            # Add injection to each SFT
            inj_SFT_filenames = []
            logger.info("Writing SFTs with injection...")
            for SFT_filename in input_SFT_filenames:

                # Load SFT and parse its filename
                SFT_catalog = lalp.SFTdataFind(SFT_filename, None)
                SFT_desc = SFT_catalog.data[0]
                SFTs = lalp.LoadMultiSFTs(SFT_catalog, -1, -1)
                SFT_fn_spec = parse_SFT_filename(SFT_filename)

                # Set up CWMakeFakeMultiData() arguments
                inj_params_v = lalp.CreatePulsarParamsVector(1)
                inj_params_v.data[0] = inj_params
                mfd_params = lalp.CWMFDataParams()
                mfd_params.fMin = SFTs.data[0].data[0].f0
                mfd_params.Band = (
                    SFTs.data[0].data[0].deltaF * SFTs.data[0].data[0].data.length
                )
                lalp.MultiLALDetectorFromMultiSFTs(mfd_params.multiIFO, SFTs)
                mfd_params.multiNoiseFloor.length = mfd_params.multiIFO.length
                mfd_params.multiTimestamps = lalp.ExtractMultiTimestampsFromSFTs(SFTs)

                # Simulate CW signal with injection parameters and add to input SFTs
                inj_SFTs = 0
                inj_SFTs, _ = lalp.CWMakeFakeMultiData(
                    inj_SFTs, None, inj_params_v, mfd_params, edat
                )
                lalp.MultiSFTVectorAdd(SFTs, inj_SFTs)
                del inj_SFTs

                # Write SFT with injection to file for later use
                if SFT_fn_spec.pubObsRun > 0:
                    SFT_fn_spec.pubChannel += "wINJECT"
                else:
                    SFT_fn_spec.privMisc += "wINJECT"
                SFT_fn_spec.path = str(inj_SFT_dir)
                inj_SFT_filename = lalp.BuildSFTFilenameFromSpec(SFT_fn_spec)
                lalp.WriteSFTVector2NamedFile(
                    SFTs.data[0],
                    inj_SFT_filename,
                    SFT_desc.window_type,
                    SFT_desc.window_param,
                    SFT_desc.comment
                    + f"\n\ngw-line-cleaner { __version__}: added injection",
                )
                logger.info("Wrote SFT with injection to file %s", inj_SFT_filename)
                inj_SFT_filenames.append(inj_SFT_filename)

            # Use SFTs with injection
            input_SFT_filenames = inj_SFT_filenames

        # Compute the F-statistic before cleaning
        if args.Fstat_before_params:
            out_stats["Fstat_before"] = compute_F_statistic(
                "before cleaning", input_SFT_filenames, args.Fstat_before_params, edat
            )

        # Compute power spectral densities (PSDs) over each detector
        PSD_freq = None
        PSD_dfreq = None
        detector_PSDs = {}
        PSD_arith_mean_to_harm_mean = {}
        SFT_catalog = lalp.SFTdataFind(";".join(input_SFT_filenames), None)
        detectors = lalp.ListIFOsInCatalog(SFT_catalog).data[:]
        logger.info(
            "Loading input SFTs from %i detectors: %s",
            len(detectors),
            ", ".join(detectors),
        )
        out_stats["num_SFTs"] = {}
        out_spectra["ASD_before"] = {}
        logger.info("Computing PSDs before cleaning")
        for det in detectors:

            # Load SFTs for the given detector
            SFT_constraints = lalp.SFTConstraints()
            SFT_constraints.detector = det
            SFT_catalog = lalp.SFTdataFind(
                ";".join(input_SFT_filenames), SFT_constraints
            )
            SFTs = lalp.LoadMultiSFTs(SFT_catalog, -1, -1)
            out_stats["num_SFTs"][det] = SFTs.data[0].length
            logger.info(
                "Computing PSD of %i SFTs from detector %s",
                out_stats["num_SFTs"][det],
                det,
            )

            # Compute PSD of the input SFTs
            SFT_f0 = SFTs.data[0].data[0].f0
            SFT_deltaF = SFTs.data[0].data[0].deltaF
            SFT_nBins = SFTs.data[0].data[0].data.length
            PSD, PSDs, _ = lalp.ComputePSDandNormSFTPower(
                inputSFTs=SFTs,
                returnMultiPSDVector=True,
                returnNormSFT=False,
                blocksRngMed=0,
                PSDmthopSFTs=lalp.MATH_OP_HARMONIC_MEAN,
                PSDmthopIFOs=0,
                nSFTmthopSFTs=0,
                nSFTmthopIFOs=0,
                normalizeByTotalNumSFTs=False,
                FreqMin=SFT_f0,
                FreqBand=SFT_deltaF * SFT_nBins,
                normalizeSFTsInPlace=False,
            )
            detector_PSDs[det] = PSD.data

            # Record ratio of PSD arithmetic mean to harmonic mean
            PSD_arith_mean = np.zeros(PSD.data.shape)
            for i in range(PSDs.data[0].length):
                PSD_arith_mean += PSDs.data[0].data[i].data.data
            PSD_arith_mean /= PSDs.data[0].length
            PSD_arith_mean_to_harm_mean[det] = PSD_arith_mean / PSD.data

            # Set/check frequencies
            PSD_freq_det = SFT_f0 + SFT_deltaF * np.arange(0, SFT_nBins)
            if PSD_freq is None:
                PSD_freq = PSD_freq_det
                PSD_dfreq = SFT_deltaF
            else:
                np.testing.assert_allclose(PSD_freq, PSD_freq_det, atol=0, rtol=1e-10)

            # Output amplitude spectral density (ASD)
            out_spectra["ASD_before"][det] = np.column_stack(
                (PSD_freq, np.sqrt(PSD.data))
            ).astype(np.float32)

        # Clean PSDs
        logger.info("Cleaning PSDs...")
        cleaned_detector_PSDs, cleaned_masks = apply_cleaning(
            PSD_freq, detector_PSDs, min_detectors=2
        )
        logger.info("Cleaning PSDs... done")

        # Find cleaned ranges and their frequencies
        cleaned_mask_ranges = {}
        cleaned_freq_ranges = {}
        cleaned_freq_ranges_str = {}
        for det in detectors:
            cleaned_mask_ranges[det] = np.argwhere(
                np.diff(cleaned_masks[det], prepend=False, append=False)
            ).reshape(-1, 2)
            cleaned_freq_ranges[det] = [
                (
                    PSD_freq[cleaned_mask_ranges[det][i, 0]],
                    PSD_freq[cleaned_mask_ranges[det][i, 1]] + PSD_dfreq,
                )
                for i in range(cleaned_mask_ranges[det].shape[0])
            ]
            cleaned_freq_ranges_str[det] = "; ".join(
                "[{:.8g}, {:.8g}]".format(*freq_range)
                for freq_range in cleaned_freq_ranges[det]
            )
            logger.info("Cleaned the following frequencies in %s data (Hz):", det)
            for line in textwrap.wrap(cleaned_freq_ranges_str[det]):
                logger.info("    %s", line)

        # Write cleaned SFTs
        logger.info("Writing cleaned SFTs")
        cleaned_SFT_filenames = []
        for SFT_filename in input_SFT_filenames:

            # Load SFT and parse its filename
            SFT_catalog = lalp.SFTdataFind(SFT_filename, None)
            SFT_desc = SFT_catalog.data[0]
            SFTs = lalp.LoadMultiSFTs(SFT_catalog, -1, -1)
            SFT_fn_spec = parse_SFT_filename(SFT_filename)

            # Compute PSDs of SFTs
            SFT_f0 = SFTs.data[0].data[0].f0
            SFT_deltaF = SFTs.data[0].data[0].deltaF
            SFT_nBins = SFTs.data[0].data[0].data.length
            _, PSDs, _ = lalp.ComputePSDandNormSFTPower(
                inputSFTs=SFTs,
                returnMultiPSDVector=True,
                returnNormSFT=False,
                blocksRngMed=0,
                PSDmthopSFTs=lalp.MATH_OP_HARMONIC_MEAN,
                PSDmthopIFOs=0,
                nSFTmthopSFTs=0,
                nSFTmthopIFOs=0,
                normalizeByTotalNumSFTs=False,
                FreqMin=SFT_f0,
                FreqBand=SFT_deltaF * SFT_nBins,
                normalizeSFTsInPlace=False,
            )

            # Apply cleaning to SFTs
            det = SFTs.data[0].data[0].name
            ii = cleaned_masks[det]
            for i in range(SFTs.data[0].length):
                SFT_i = SFTs.data[0].data[i].data
                PSD_i = PSDs.data[0].data[i].data

                # Clean SFT
                clean_ii = cleaned_detector_PSDs[det][ii] / PSD_i.data[ii]
                SFT_i.data[ii] *= np.sqrt(clean_ii)

                # Correct harmonic mean bias
                harm_mean_bias_ii = np.median(
                    np.sqrt(PSD_arith_mean_to_harm_mean[det][ii])
                )
                SFT_i.data[ii] *= harm_mean_bias_ii

            # Write cleaned SFT to file for output
            if SFT_fn_spec.pubObsRun > 0:
                SFT_fn_spec.pubChannel += "wGWLINECLEAN"
            else:
                SFT_fn_spec.privMisc += "wGWLINECLEAN"
            SFT_fn_spec.path = str(args.output_SFT_dir)
            cleaned_SFT_filename = lalp.BuildSFTFilenameFromSpec(SFT_fn_spec)
            lalp.WriteSFTVector2NamedFile(
                SFTs.data[0],
                cleaned_SFT_filename,
                SFT_desc.window_type,
                SFT_desc.window_param,
                SFT_desc.comment
                + f"\n\ngw-line-cleaner { __version__}: cleaned frequencies (Hz):\n{cleaned_freq_ranges_str[det]}",
            )
            logger.info("Wrote cleaned SFT file %s", cleaned_SFT_filename)
            cleaned_SFT_filenames.append(cleaned_SFT_filename)

            # Check if injection intersected with a cleaned line
            if args.inject_params:
                Tsft = 1.0 / SFTs.data[0].data[0].deltaF
                det_states = lalp.GetMultiDetectorStatesFromMultiSFTs(
                    SFTs, edat, 0.5 * Tsft
                )
                inj_f_min, inj_f_max = lalp.CWSignalBand(
                    det_states.data[0], inj_params.Doppler
                )
                if det not in out_stats["inj_intersect_cleaned"]:
                    out_stats["inj_intersect_cleaned"][det] = False
                if any(
                    inj_f_min <= f_max and f_min <= inj_f_max
                    for f_min, f_max in cleaned_freq_ranges[det]
                ):
                    out_stats["inj_intersect_cleaned"][det] = True

        # Compute the F-statistic after cleaning
        if args.Fstat_after_params:
            out_stats["Fstat_after"] = compute_F_statistic(
                "after cleaning", cleaned_SFT_filenames, args.Fstat_after_params, edat
            )

        # Compute PSDs after cleaning
        out_spectra["ASD_after"] = {}
        logger.info("Computing PSDs after cleaning")
        for det in detectors:

            # Load SFTs for the given detector
            SFT_constraints = lalp.SFTConstraints()
            SFT_constraints.detector = det
            SFT_catalog = lalp.SFTdataFind(
                ";".join(cleaned_SFT_filenames), SFT_constraints
            )
            SFTs = lalp.LoadMultiSFTs(SFT_catalog, -1, -1)
            out_stats["num_SFTs"][det] = SFTs.data[0].length
            logger.info(
                "Computing PSD of %i SFTs from detector %s",
                out_stats["num_SFTs"][det],
                det,
            )

            # Compute PSD of the cleaned SFTs
            SFT_f0 = SFTs.data[0].data[0].f0
            SFT_deltaF = SFTs.data[0].data[0].deltaF
            SFT_nBins = SFTs.data[0].data[0].data.length
            PSD, PSDs, _ = lalp.ComputePSDandNormSFTPower(
                inputSFTs=SFTs,
                returnMultiPSDVector=True,
                returnNormSFT=False,
                blocksRngMed=0,
                PSDmthopSFTs=lalp.MATH_OP_HARMONIC_MEAN,
                PSDmthopIFOs=0,
                nSFTmthopSFTs=0,
                nSFTmthopIFOs=0,
                normalizeByTotalNumSFTs=False,
                FreqMin=SFT_f0,
                FreqBand=SFT_deltaF * SFT_nBins,
                normalizeSFTsInPlace=False,
            )

            # Check frequencies
            PSD_freq_det = SFT_f0 + SFT_deltaF * np.arange(0, SFT_nBins)
            np.testing.assert_allclose(PSD_freq, PSD_freq_det, atol=0, rtol=1e-10)

            # Output amplitude spectral density (ASD)
            out_spectra["ASD_after"][det] = np.column_stack(
                (PSD_freq, np.sqrt(PSD.data))
            ).astype(np.float32)

        # Write output statistics
        if args.output_spectra:
            out_stats.update(out_spectra)
        if args.output_stats:
            with h5py.File(args.output_stats, "w") as f:
                write_out_stats_to_h5(f, out_stats)
            logger.info("Wrote output statistics file %s", args.output_stats)

        # Plot output spectra
        if args.output_spectra_plots:
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(ncols=2, figsize=(16, 9))
            for det, ax in zip(detectors, axs):
                ax.loglog(
                    out_spectra["ASD_before"][det][:, 0],
                    out_spectra["ASD_before"][det][:, 1],
                    "r:",
                    label="Before cleaning",
                )
                ax.loglog(
                    out_spectra["ASD_after"][det][:, 0],
                    out_spectra["ASD_after"][det][:, 1],
                    "b-",
                    label="After cleaning",
                )
                ax.grid(True)
                ax.legend(loc="best")
                ax.set_xlabel(r"Frequency / $\text{Hz}$")
                ax.set_ylabel(r"Amplitude Spectral Density / $\text{Hz}^{-1/2}$")
                ax.set_title(f"Detector: {det}")
            fig.savefig(args.output_spectra_plots, dpi=72)
            logger.info("Plotted output spectra to file %s", args.output_spectra_plots)

    logger.info("DONE")

    return 0
