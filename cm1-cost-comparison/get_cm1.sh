#!/usr/bin/env bash
# Fetch, patch, and build CM1 release 19.6 for the RCEMIP cost benchmark.
#
# cm1r19.6 is the exact release used for the RCEMIP CM1 submission (see the
# model documentation form cited in README.md). The patch activates the
# RCEMIP configuration values that ship inside the release as commented
# "!!! ... ! rcemip" lines; it adds nothing that is not already in the code.
set -euo pipefail
cd "$(dirname "$0")"

URL=https://www2.mmm.ucar.edu/people/bryan/cm1/cm1r19.6.tar.gz
SHA256=68ac7657d96660d00e395e0e9229a85dc6d40174439f02d9b57539788d490745

# ---- machine-specific build configuration: edit these five lines ----------
MPIF90=/usr/lib64/openmpi/bin/mpif90                # MPI Fortran wrapper
NETCDF_INC=/usr/lib64/gfortran/modules/openmpi     # dir with netcdf.mod
NETCDF_LIB=/usr/lib64/openmpi/lib                  # dir with libnetcdff
FFLAGS="-ffree-form -ffree-line-length-none -O2 -finline-functions -march=native"
COMPAT="-fallow-argument-mismatch -std=legacy"     # needed for gfortran >= 10
# ---------------------------------------------------------------------------

[ -f cm1r19.6.tar.gz ] || curl -fsSLO "$URL"
echo "$SHA256  cm1r19.6.tar.gz" | sha256sum -c -

rm -rf cm1r19
tar xzf cm1r19.6.tar.gz
patch -p1 -d cm1r19 < rcemip_activate.patch

cat > cm1r19/src/Makefile.local <<EOF
OUTPUTOPT = -DNETCDF -DNCFPLUS
OUTPUTINC = -I$NETCDF_INC
OUTPUTLIB = -L$NETCDF_LIB
LINKOPTS  = -lnetcdf -lnetcdff
FC   = $MPIF90
OPTS = $FFLAGS $COMPAT
CPP  = cpp -C -P -traditional -Wno-invalid-pp-token -ffreestanding
DM   = -DMPI
EOF
# Inject the local config at the top of the shipped Makefile.
sed -i '1i include Makefile.local' cm1r19/src/Makefile

# The shipped Makefile has incomplete module dependencies; build serially.
make -C cm1r19/src
echo "Built: cm1r19/run/cm1.exe"
