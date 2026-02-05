#!/usr/bin/env python3
import argparse
import numpy as np
import h5py

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5file")
    ap.add_argument("--wmin", type=float, default=40000.0, help="Angstrom (4 micron)")
    ap.add_argument("--wmax", type=float, default=50000.0, help="Angstrom (5 micron)")
    ap.add_argument("--out", default=None, help="Optional .npz output")
    args = ap.parse_args()

    with h5py.File(args.h5file, "r") as h5:
        mode = h5["spectra"].attrs.get("mode", "unknown")
        if isinstance(mode, (bytes, bytearray)):
            mode = mode.decode("utf-8", "replace")
        if mode != "common_grid":
            raise SystemExit(f"ERROR: mode={mode}, expected common_grid")

        w = h5["spectra/wavelength"][:]   # (nlam,)
        lo, hi = args.wmin, args.wmax
        i0 = int(np.searchsorted(w, lo, side="left"))
        i1 = int(np.searchsorted(w, hi, side="right"))
        w_sub = w[i0:i1]

        # This is the key: read all gridpoints for the wavelength slice
        flux_sub = h5["spectra/flux"][:, i0:i1]  # (ngp, nlam_sub)

    print(f"Loaded flux array: {flux_sub.shape} covering {w_sub.min():.2f}..{w_sub.max():.2f} A")

    if args.out:
        np.savez_compressed(args.out, wavelength_angstrom=w_sub, flux=flux_sub)
        print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()