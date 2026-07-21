#!/usr/bin/env python3
"""Small-domain (RCE_small_les300) version of the RCEMIP-deltas pipeline.

Four models have small-domain data: SAM (native, 480^2 @ 200 m, QN combined
condensate split by steam's lambda(T)), CM1 (540^2), DALES (504^2, no
pressure), ICON-LEM (500^2, 156 levels, top-down index z via FMSE inversion).
Last 10 daily snapshots each (ICON has only 4: days 25-28 -- all used).

STEAM: same frozen config as the channel comparison but on the small-domain
geometry: 512^2 @ 200 m (102.4 km = one outer-scale tile), ls = 10 m,
20 km top, seed = 3000 + snapshot. Each run is reduced immediately (profile
stats, 4/10 km anomaly samples, tau > 1 mask via cloudyview liquid+ice) and
the .nc deleted.

Phases (args, default all): hosts steam figures
Outputs: stats_small/<case>_snap<i>.npz, figs/{deltas,pdfs,fractal}_small.png
"""

import importlib.util
import sys
from pathlib import Path

import netCDF4
import numpy as np

from steam.constants import (
    specific_heat_dry_air as cp,
    latent_heat_vaporization as Lv,
    gravity as g,
)

HERE = Path(__file__).parent
DATA = HERE / "data"
STATS = HERE / "stats_small"
RUNS = HERE / "runs"
FIGS = HERE / "figs"
CLOUD_TAU = 1.0
DX = 200.0
PDF_LEVELS_M = (4000.0, 10000.0)
PROFILE_DZ = 50.0
DOMAIN_HEIGHT = 20000.0
P0_DEFAULT = 101480.0
N_SNAPS = {"sam": 10, "cm1": 10, "dales": 10, "icon_lem": 4}

spec = importlib.util.spec_from_file_location(
    "cv_optical_depth",
    Path.home() / "code-and-data/cloudyview/cloudyview/optical_depth.py")
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


def spec_to_mr(q):
    return q / (1.0 - q)


def read_var(path, name, index=None):
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    v = ds.variables[name]
    data = v[index] if index is not None else v[:]
    ds.close()
    return np.asarray(data)


# ── host adapters -> (z, T, qv_mr, qc_mr, qi_mr, p_surf), (nz, ny, nx) ──────


def sam(i):
    files = sorted((DATA / "sam_small").glob("RCEMIP_SST300_*.nc"))
    path = files[i]
    z = read_var(path, "z")
    T = read_var(path, "TABS", 0)
    qv = read_var(path, "QV", 0) / 1000.0        # g/kg mixing ratio
    cond = read_var(path, "QN", 0) / 1000.0      # combined liquid+ice
    lam = np.clip((T - 235.15) / (273.15 - 235.15), 0.0, 1.0)
    p_ref_mb = read_var(path, "p")
    return z, T, qv, lam * cond, (1.0 - lam) * cond, float(p_ref_mb[0] * 100.0)


def cm1(i):
    files = sorted((DATA / "cm1_small").glob("CM1_RCE_small_les300_*.nc"))
    path = files[i]
    z = read_var(path, "z")
    T = read_var(path, "ta", 0)
    qv = spec_to_mr(read_var(path, "hus", 0))
    qc = spec_to_mr(read_var(path, "clw", 0))
    qi = spec_to_mr(read_var(path, "cli", 0))
    pa = read_var(path, "pa", 0)
    return z, T, qv, qc, qi, float(pa[0].mean(dtype=np.float64))


def dales(i):
    d = DATA / "dales_small"
    t = 21 + i  # last 10 of 31 daily snapshots
    z = read_var(d / "DALES_RCE_small_les300_3D_ta.nc", "zt")
    T = read_var(d / "DALES_RCE_small_les300_3D_ta.nc", "ta", t)
    qv = spec_to_mr(read_var(d / "DALES_RCE_small_les300_3D_hus.nc", "hus", t))
    qc = spec_to_mr(read_var(d / "DALES_RCE_small_les300_3D_clw.nc", "clw", t))
    qi = spec_to_mr(read_var(d / "DALES_RCE_small_les300_3D_cli.nc", "cli", t))
    for name, a in (("ta", T), ("hus", qv), ("clw", qc), ("cli", qi)):
        assert not np.any(a == -999.0), f"dales {name}: fill values present"
    return z, T, qv, qc, qi, P0_DEFAULT


def icon_lem(i):
    d = DATA / "icon_lem_small"
    main = d / "ICON_LEM_CRM-RCE_small_les_300-3D_last25d.nc"
    T = read_var(main, "ta", i)
    hus = read_var(main, "hus", i)
    clw = read_var(main, "clw", i)
    cli = read_var(main, "cli", i)
    pa = read_var(main, "pa", i)
    fmse = read_var(d / "ICON_LEM_CRM-RCE_small_les_300-3D_FMSE.nc",
                    "fmse_3d", 0)
    T0 = read_var(main, "ta", 0)
    hus0 = read_var(main, "hus", 0)
    cli0 = read_var(main, "cli", 0)
    zf = (fmse - 1004.64 * T0 - 2500800.0 * hus0 + 333700.0 * cli0) / 9.80665
    z = zf.mean(axis=(1, 2), dtype=np.float64)
    assert np.all(zf.std(axis=(1, 2), dtype=np.float64) < 50.0), "zf not flat"
    p_surf = float(pa[-1].mean(dtype=np.float64))
    flip = lambda a: a[::-1]
    return (flip(z), flip(T), flip(spec_to_mr(hus)), flip(spec_to_mr(clw)),
            flip(spec_to_mr(cli)), p_surf)


ADAPTERS = {"sam": sam, "cm1": cm1, "dales": dales, "icon_lem": icon_lem}


# ── shared reduction: fields -> npz (works for hosts and STEAM) ─────────────


def reduce_fields(z, T_or_none, h3, qt3, qc3, qi3, p_surf, out_path,
                  with_profile):
    """h3/qt3/qc3/qi3 shaped (nz, ny, nx) float; z 1D increasing [m]."""
    nz = z.size
    h_mean = np.empty(nz)
    h_var = np.empty(nz)
    qt_mean = np.empty(nz)
    qt_var = np.empty(nz)
    cloud_fraction = np.empty(nz)
    for k in range(nz):
        hk = h3[k].astype(np.float64)
        qtk = qt3[k].astype(np.float64)
        condk = qc3[k].astype(np.float64) + qi3[k].astype(np.float64)
        h_mean[k] = hk.mean()
        h_var[k] = hk.var()
        qt_mean[k] = qtk.mean()
        qt_var[k] = qtk.var()
        cloud_fraction[k] = np.mean(condk > 0.01e-3)

    samples = {}
    for target in PDF_LEVELS_M:
        k = int(np.argmin(np.abs(z - target)))
        samples[f"h_anom_{target:.0f}"] = (
            h3[k].astype(np.float64) - h_mean[k]).ravel().astype(np.float32)
        samples[f"qt_anom_{target:.0f}"] = (
            qt3[k].astype(np.float64) - qt_mean[k]).ravel().astype(np.float32)

    lwc = np.moveaxis(qc3, 0, -1) * 1000.0
    iwc = np.moveaxis(qi3, 0, -1) * 1000.0
    tau = cv.vertically_integrated_optical_depth(
        lwc, np.asarray(z, dtype=np.float64), iwc=iwc)
    mask = (tau > CLOUD_TAU).astype(np.uint8)

    extra = {}
    if with_profile:
        z_uniform = np.arange(0.0, DOMAIN_HEIGHT + PROFILE_DZ, PROFILE_DZ)
        extra = dict(z_profile=z_uniform,
                     h_profile=np.interp(z_uniform, z, h_mean),
                     qt_profile=np.interp(z_uniform, z, qt_mean),
                     surface_pressure=p_surf)
    np.savez(out_path, z=z, h_mean=h_mean, h_var=h_var, qt_mean=qt_mean,
             qt_var=qt_var, cloud_fraction=cloud_fraction, mask=mask,
             **samples, **extra)


def hosts():
    STATS.mkdir(exist_ok=True)
    for model, adapter in ADAPTERS.items():
        for i in range(N_SNAPS[model]):
            z, T, qv, qc, qi, p_surf = adapter(i)
            z = np.asarray(z, dtype=np.float64)
            assert np.all(np.diff(z) > 0), f"{model}: z not increasing"
            for name, a in (("T", T), ("qv", qv), ("qc", qc), ("qi", qi)):
                bad = ~np.isfinite(a) | (np.abs(a) > 1e8)
                assert not bad.any(), f"{model} snap{i}: bad {name}"
            h3 = (cp * T.astype(np.float64)
                  + g * z[:, None, None] + Lv * qv.astype(np.float64))
            qt3 = qv.astype(np.float64) + qc + qi
            reduce_fields(z, T, h3, qt3, qc, qi, p_surf,
                          STATS / f"{model}_snap{i}.npz", with_profile=True)
            print(f"host {model} snap{i} done")


def steam_runs():
    from steam.simulate import simulate
    from steam.thermodynamics import compute_diagnostics
    RUNS.mkdir(exist_ok=True)
    for model in ADAPTERS:
        for i in range(N_SNAPS[model]):
            src = np.load(STATS / f"{model}_snap{i}.npz")
            out_nc = RUNS / f"steam_small_{model}_snap{i}.nc"
            h_profile = src["h_profile"]
            qt_profile = src["qt_profile"]
            simulate(
                h_profile, qt_profile,
                nx=512, ny=512, dx=DX, dy=DX,
                outer_scale=102400.0,
                spheroscale=np.full_like(src["z_profile"], 10.0),
                anisotropy="piecewise_isotropic_below_spheroscale",
                domain_height=DOMAIN_HEIGHT,
                profile_dz=PROFILE_DZ,
                output_path=str(out_nc),
                surface_pressure=float(src["surface_pressure"]),
                seed=3000 + i,
                h_min=h_profile.min() - 10 * cp,
                h_max=h_profile.max() + 10 * cp,
                qt_min=0.0, qt_max=max(0.03, 1.5 * qt_profile.max()),
                compress=True,
                device="cuda",
            )
            compute_diagnostics(str(out_nc), compress=True)
            ds = netCDF4.Dataset(out_nc)
            ds.set_auto_mask(False)
            z = ds.variables["z"][:].astype(np.float64)
            # STEAM stores (x, y, z); reduce_fields wants (nz, ny, nx).
            h3 = np.moveaxis(ds.variables["h"][:], -1, 0)
            qt3 = np.moveaxis(ds.variables["qt"][:], -1, 0)
            qc3 = np.moveaxis(ds.variables["qc"][:], -1, 0)
            qi3 = np.moveaxis(ds.variables["qi"][:], -1, 0)
            ds.close()
            reduce_fields(z, None, h3, qt3, qc3, qi3, None,
                          STATS / f"steam_{model}_snap{i}.npz",
                          with_profile=False)
            out_nc.unlink()
            print(f"steam {model} snap{i} done")


# ── figures ─────────────────────────────────────────────────────────────────


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import objscale

    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": "#B9B3AC", "axes.linewidth": 0.8,
        "grid.color": "#E5E1DC", "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200,
    })
    models = list(ADAPTERS)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    Z = np.arange(0.0, 20000.0 + 100.0, 100.0)
    QUANTITIES = [
        ("cloud_fraction", lambda d: d["cloud_fraction"], "cloud fraction"),
        ("h_std", lambda d: np.sqrt(d["h_var"]) / cp, "std $h'/c_p$ [K]"),
        ("qt_std", lambda d: np.sqrt(d["qt_var"]) * 1e3, "std $q_t'$ [g/kg]"),
    ]

    def snap_mean(prefix, n):
        out = {}
        for key, tr, _ in QUANTITIES:
            curves = []
            for i in range(n):
                d = np.load(STATS / f"{prefix}_snap{i}.npz")
                curves.append(np.interp(Z, d["z"], tr(d),
                                        left=np.nan, right=np.nan))
            out[key] = np.nanmean(curves, axis=0)
        return out

    host = {m: snap_mean(m, N_SNAPS[m]) for m in models}
    steam = {m: snap_mean(f"steam_{m}", N_SNAPS[m]) for m in models}

    # deltas with pairwise bands
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.8), sharey=True)
    for ax, (key, _, label) in zip(axes, QUANTITIES):
        ensemble = np.array([host[m][key] for m in models])
        n = len(models)
        pairs = np.array([ensemble[a] - ensemble[b]
                          for a in range(n) for b in range(n) if a != b])
        ax.fill_betweenx(Z / 1000, np.nanmin(pairs, axis=0),
                         np.nanmax(pairs, axis=0), color="#E9E6E1",
                         label="LES$-$LES full range", lw=0)
        ax.fill_betweenx(Z / 1000, np.nanpercentile(pairs, 25, axis=0),
                         np.nanpercentile(pairs, 75, axis=0), color="#CFCAC2",
                         label="LES$-$LES 25$-$75%", lw=0)
        for m, c in zip(models, colors):
            ax.plot(steam[m][key] - host[m][key], Z / 1000, color=c, lw=1.2,
                    label=m)
        ax.axvline(0, color="#9A938B", lw=0.8)
        ax.set(xlabel=f"$\\Delta$ {label}", ylim=(0, 20))
        ax.grid(True, alpha=0.6)
    axes[0].set_ylabel("z [km]")
    axes[2].legend(fontsize=6.5)
    fig.suptitle("small LES: STEAM $-$ host, vs inter-LES spread", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "deltas_small.png")
    plt.close(fig)

    # anomaly PDFs
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    for row, name, symbol, scale, unit in [
        (0, "qt", "q_t'", 1000.0, "g/kg"),
        (1, "h", "h'/c_p", 1.0 / cp, "K"),
    ]:
        for col, target in enumerate(PDF_LEVELS_M):
            ax = axes[row, col]
            for m, c in zip(models, colors):
                for prefix, ls in ((m, "-"), (f"steam_{m}", "--")):
                    pooled = np.concatenate([
                        np.load(STATS / f"{prefix}_snap{i}.npz")[
                            f"{name}_anom_{target:.0f}"]
                        for i in range(N_SNAPS[m])]) * scale
                    density, edges = np.histogram(pooled, bins=150,
                                                  density=True)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.plot(centers, density, color=c, lw=1.0, ls=ls,
                            label=m if ls == "-" else None)
            ax.set_yscale("log")
            ax.set_title(f"${symbol}$  at  z $\\approx$ {target / 1000:.0f} km")
            ax.set_xlabel(f"${symbol}$ [{unit}]")
            ax.grid(True, alpha=0.5)
            if col == 0:
                ax.set_ylabel("PDF")
    axes[0, 1].legend(fontsize=6.5, title="solid host / dash STEAM",
                      title_fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / "pdfs_small.png")
    plt.close(fig)

    # fractal: correlation dimension + area distributions
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for m, c in zip(models, colors):
        results = {}
        for prefix, ls in ((m, "-"), (f"steam_{m}", "--")):
            masks = [np.load(STATS / f"{prefix}_snap{i}.npz")["mask"]
                     .astype(np.float32) for i in range(N_SNAPS[m])]
            sizes = np.full(masks[0].shape, DX)
            dim, bins, C_l = objscale.ensemble_correlation_dimension(
                masks, x_sizes=sizes, y_sizes=sizes,
                point_reduction_factor=10, return_C_l=True)
            exponent, (lb, lc) = objscale.finite_array_powerlaw_exponent(
                masks, "area", x_sizes=sizes, y_sizes=sizes,
                return_counts=True)
            results[prefix] = (dim, bins, C_l, exponent, lb, lc)
        for prefix, ls in ((m, "-"), (f"steam_{m}", "--")):
            dim, bins, C_l, exponent, lb, lc = results[prefix]
            d_label = (f"{m} {results[m][0]:.2f}/"
                       f"{results[f'steam_{m}'][0]:.2f}") if ls == "-" else None
            s_label = (f"{m} {results[m][3]:.2f}/"
                       f"{results[f'steam_{m}'][3]:.2f}") if ls == "-" else None
            axes[0].loglog(bins / 1000.0, C_l, color=c, lw=1.1, ls=ls,
                           label=d_label)
            axes[1].plot(lb - 6.0, lc, color=c, lw=1.1, ls=ls, label=s_label)
        print(f"{m}: D2 {results[m][0]:.2f}/{results[f'steam_{m}'][0]:.2f}, "
              f"area {results[m][3]:.2f}/{results[f'steam_{m}'][3]:.2f}")
    axes[0].set(xlabel="r [km]", ylabel="correlation integral $C(r)$",
                title="$\\tau>1$ correlation integral (host/STEAM $D_2$)")
    axes[1].set(xlabel="log$_{10}$ area [km$^2$]", ylabel="log$_{10}$ counts",
                title="area distribution (host/STEAM slope)")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.5)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fractal_small.png")
    plt.close(fig)
    print("wrote figs/deltas_small.png, figs/pdfs_small.png, "
          "figs/fractal_small.png")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["hosts", "steam", "figures"]
    if "hosts" in wanted:
        hosts()
    if "steam" in wanted:
        steam_runs()
    if "figures" in wanted:
        figures()
