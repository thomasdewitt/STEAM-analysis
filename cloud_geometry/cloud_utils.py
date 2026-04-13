"""Shared utilities for cloud geometry analysis scripts."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIGURES_DIR = Path(__file__).resolve().parent.parent / 'Figures'

DATASET_COLORS = {
    'SAM_TWPICE': '#6a51a3',   # purple
    'SAM_RCEMIP': '#2171b5',   # blue
    'CM1':        '#e6550d',   # orange
    'STEAM':      '#238b45',   # green
}

DATASET_LABELS = {
    'SAM_TWPICE': 'SAM TWPICE',
    'SAM_RCEMIP': 'SAM RCEMIP\n(RCE_small_les300)',
    'CM1':        'CM1\n(RCE_small_les300)',
    'STEAM':      'STEAM',
}

DATASETS = ['SAM_TWPICE', 'SAM_RCEMIP', 'CM1', 'STEAM']


def _load_binary_arrays(dataset, tau_threshold=1.0, experiment='RCE_small_les300'):
    """Load binary cloud-mask arrays from a dataset via optical depth thresholding.

    Processes one timestep at a time to keep memory usage low: loads a single
    3D LWC (+IWC) slice, computes optical depth, binarizes, then discards the
    3D data before loading the next timestep.

    Parameters
    ----------
    dataset : str
        One of 'SAM_TWPICE', 'SAM_RCEMIP', 'CM1', 'STEAM'.
    tau_threshold : float
        Optical depth threshold for binarization (default 1.0).
    experiment : str
        Experiment name for SAM_RCEMIP and CM1 (default 'RCE_small_les300').

    Returns
    -------
    binary_arrays : list of 2D bool ndarray
        One array per timestep, shape (nx, ny).
    dx : float
        Horizontal grid spacing in metres.
    label : str
        Human-readable dataset label.
    """
    from cloudyview import vertically_integrated_optical_depth
    from config import (SAM_TWPICE_TIMESTEPS, SAM_RCEMIP_SMALL_TIMESTEPS,
                        CM1_SMALL_TIMESTEPS, STEAM_DATA_DIR, STEAM_FILE_PATTERN)

    # Build an iterable of (lwc_loader_kwargs, iwc_loader_kwargs_or_None)
    # so we can call each loader once per timestep.
    if dataset == 'SAM_TWPICE':
        from utils.sam_twpice_loader import load_sam_twpice_variable
        timesteps = SAM_TWPICE_TIMESTEPS

        def _load_one(ts):
            z, lwc, dx = load_sam_twpice_variable('QC', timesteps=[ts], horiz_stride=1)
            _, iwc, _  = load_sam_twpice_variable('QI', timesteps=[ts], horiz_stride=1)
            return z, lwc[0], iwc[0], dx

        slices = timesteps

    elif dataset == 'SAM_RCEMIP':
        from utils.sam_rcemip_loader import load_sam_rcemip_variable
        timesteps = SAM_RCEMIP_SMALL_TIMESTEPS

        def _load_one(ts):
            z, lwc, dx = load_sam_rcemip_variable(
                'QN', experiment=experiment, timesteps=[ts], horiz_stride=1)
            return z, lwc[0], None, dx

        slices = timesteps

    elif dataset == 'CM1':
        from utils.cm1_loader import load_cm1_variable
        timesteps = CM1_SMALL_TIMESTEPS

        def _load_one(ts):
            z, lwc, dx = load_cm1_variable(
                'clw', experiment=experiment, timesteps=[ts], horiz_stride=1)
            _, iwc, _  = load_cm1_variable(
                'cli', experiment=experiment, timesteps=[ts], horiz_stride=1)
            return z, lwc[0], iwc[0], dx

        slices = timesteps

    elif dataset == 'STEAM':
        from utils.steam_loader import load_steam_variable
        seed_files = sorted(Path(STEAM_DATA_DIR).glob(STEAM_FILE_PATTERN))
        if not seed_files:
            raise FileNotFoundError(
                f"No STEAM files matching '{STEAM_FILE_PATTERN}' in {STEAM_DATA_DIR}")

        def _load_one(fpath):
            z, lwc, dx = load_steam_variable('qc', seed_files=[fpath], horiz_stride=1)
            _, iwc, _  = load_steam_variable('qi', seed_files=[fpath], horiz_stride=1)
            return z, lwc[0], iwc[0], dx

        slices = seed_files

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from {DATASETS}.")

    import gc

    binary_arrays = []
    dx = None

    for i, s in enumerate(slices):
        print(f"  [{dataset}] slice {i + 1}/{len(slices)}: loading and computing τ...")
        z, lwc_3d, iwc_3d, dx = _load_one(s)
        z = np.asarray(z, dtype=float)
        # Downcast to float32 so cloudyview works on the smaller type.
        # copy=False avoids an extra allocation when the loader already returns float32
        # (SAM_TWPICE, factor=1 int); for float64 loaders (CM1, STEAM, SAM_RCEMIP)
        # this halves the memory footprint before tau is computed.
        lwc_3d = lwc_3d.astype(np.float32, copy=False)
        if iwc_3d is not None:
            iwc_3d = iwc_3d.astype(np.float32, copy=False)
        print(f"    shape={lwc_3d.shape}, dtype={lwc_3d.dtype}, "
              f"~{lwc_3d.nbytes / 1e6:.0f} MB (lwc)")
        tau = vertically_integrated_optical_depth(lwc_3d, z, iwc=iwc_3d)
        # Explicitly release the 3D arrays before the next load
        del lwc_3d, iwc_3d
        gc.collect()
        binary_arrays.append(tau > tau_threshold)
        del tau

    return binary_arrays, dx, DATASET_LABELS[dataset]
