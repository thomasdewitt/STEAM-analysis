# CM1 vs STEAM: computational cost on one machine

This folder measures the wall-clock cost of an RCEMIP channel simulation and
compares it to STEAM realizations on the same hardware. It backs the cost
comparison in the paper supplement.

## Provenance chain

1. The RCEMIP CM1 submission used **CM1 release 19.6**, per the model
   documentation form on the DKRZ archive
   (`RCEMIP/CM1/rcemip_modeldoc_form_CM1.pdf`; Bryan, NCAR). That form also
   documents the RCE_large grid (2016 x 134 x 74 at 3 km), the physics
   choices, and the ~18 s average time step.
2. `get_cm1.sh` downloads that exact release from the CM1 website
   (sha256-pinned) and applies `rcemip_activate.patch`.
3. The patch only activates configuration values already present in the
   release as commented `!!! ... ! rcemip` lines: RCEMIP insolation
   (551.58 W m-2, zenith 42.05 deg, albedo 0.07), the Wing et al. (2018)
   Eq. 1 ozone profile, and the RCEMIP trace gases (CO2 348 ppmv, CH4
   1650 ppbv, N2O 306 ppbv, halocarbons zero). It adds no new code.
4. `run/namelist.input` configures RCE_large300 following the model doc form
   and Wing et al. (2018): analytic sounding (isnd=21, q0 = 18.65 g/kg,
   SST = 300 K), RRTMG radiation, Morrison microphysics, Bryan-Rotunno PBL,
   revised-WRF surface layer, no rotation, sponge above 25 km.
   `run/input_grid_z` is Wing et al. (2018) Table 3, verbatim.

Known departures from the production run are listed at the bottom.

## Running it

```bash
./get_cm1.sh                                   # edit the 5 machine lines first
../.venv/bin/python benchmark.py --ranks 16
```

`benchmark.py` runs one model hour of CM1 (fixed dt = 18 s, 200 steps) and
times two single STEAM realizations on the same channel footprint, CPU only:
the paper's matched configuration (1024 x 64 x 78 at 6 km) and a host-native
one (2048 x 128 x 115 at 3 km, more grid points than CM1's own grid).
Results go to `results.json`; the CM1 component timing table (which shows
the radiation share) is kept in `cm1_benchmark.log`.

## The derived metric

The supplement quotes wall-seconds per independent 3D field. A production
channel run is 100 days, and the paper's analysis takes three independent
snapshots from it (supplement, comparison-data subsection). So CM1's cost
per independent field is (wall per model hour) x 2400 / 3. Each STEAM
realization is one independent field.

## Known departures from the production RCEMIP run

- Cold start from the analytic sounding rather than from an RCE_small
  equilibrium profile. The benchmark hour is quiescent; convecting-state
  microphysics would add a few percent to the CM1 leg.
- dtrad = 300 s comes from the RCE test case shipped with CM1; the value
  used in production is not stated in the model doc form.
- Physical constants are CM1 defaults, as shipped in the public release.
- Compiler and machine differ from the NCAR production environment.

Machine for the numbers in the supplement: AMD Ryzen 9950X (16 cores),
60 GB RAM, gfortran 15 + OpenMPI, Fedora 44. Both models use all 16 cores
(CM1 via MPI ranks, STEAM via torch CPU threads).
