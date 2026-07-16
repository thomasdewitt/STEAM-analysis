# SAM mean profiles for STEAM initialization

Horizontal-mean moist static energy h [J/kg] and total water qt [kg/kg]
profiles extracted from SAM LES output by `extract_profiles.py`, used as
STEAM's `<Phi>_t(z)` inputs. Sources on the Expansion drive are read-only.

Every horizontal reduction of a K-scale float32 field (MSE, TABS) uses an
explicit float64 accumulator: at ~4M samples per level the float32 partial
sums reach ~2e8 where ULP is O(10), which corrupts the profile by 1-2 K.

## twpice_mean_profiles.nc

- Source: `SAM-TWPICE/OUT_3D.{MSE,QV,QC,QI,PP}/TWPICE_LPT_3D_*_<stamp>.nc`,
  2048 x 2048 x 255 (stretched z to 26.65 km), dx = 100 m.
- Snapshots: stamps 0000000150, 0000001800, 0000003450 (timestep x 2 s;
  5-min cadence across the ~1.9 h archived window at day 20). Snapshots are
  highly correlated, so three spread across the window suffice.
- h = cp * MSE (file stores h/cp in K); qt = QV + QC + QI (g/kg -> kg/kg).
  Horizontal mean per level, averaged over the three snapshots.
- Reference pressure: `pres` from the PP file header (mb -> Pa);
  `surface_pressure` attribute = pres at the lowest level.

## gate_mean_profiles.nc

- Source: `SAM-GATE/com3D/GATE_IDEAL_S_2048x2048x256_100m_2s_2048_<stamp>.nc`,
  2048 x 2048 x 256 (z levels 25 -> 26979.5 m), dx = 100 m. Provenance in
  the drive's `DATA_RECORD.md` (Khairoutdinov et al. 2009 GATE idealized LES).
- Snapshots: the 12 hourly stamps with time >= 11 h (hours 11-19, 21-23;
  hour 20 is absent from the hourly NetCDF set), i.e. developed convection.
- GATE stores TABS, not MSE: h = cp * TABS + g * z + Lv * QV;
  qt = QV + QN (vapor + non-precipitating condensate), g/kg -> kg/kg.
  Horizontal mean per level with float64 accumulation, averaged over the
  12 snapshots.
- Reference pressure: per-level `p` variable (mb -> Pa).
