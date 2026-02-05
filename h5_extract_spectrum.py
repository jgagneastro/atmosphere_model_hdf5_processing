#!/usr/bin/env python3
import argparse
import numpy as np
import h5py


def _find_index_of_gridpoint(h5: h5py.File, gridpoint_id: int) -> int:
    gp_ids = h5["gridpoints/model_gridpoint_id"][:]
    # gp_ids is int32 array
    hits = np.where(gp_ids == gridpoint_id)[0]
    if hits.size == 0:
        raise SystemExit(f"ERROR: model_gridpoint_id={gridpoint_id} not found in file.")
    return int(hits[0])


def _load_common_grid(h5: h5py.File, i: int, wmin: float | None, wmax: float | None):
    w = h5["spectra/wavelength"][:]  # float64
    f = h5["spectra/flux"][i, :]     # float32

    if wmin is None and wmax is None:
        return w, f

    lo = -np.inf if wmin is None else wmin
    hi = np.inf if wmax is None else wmax
    m = (w >= lo) & (w <= hi)
    return w[m], f[m]


def _load_ragged(h5: h5py.File, i: int, wmin: float | None, wmax: float | None):
    offsets = h5["spectra/offsets"][:]
    lengths = h5["spectra/lengths"][:]
    wcat = h5["spectra/wavelength_concat"]
    fcat = h5["spectra/flux_concat"]

    off = int(offsets[i])
    n = int(lengths[i])
    if n <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float32)

    w = wcat[off:off + n]
    f = fcat[off:off + n]

    if wmin is None and wmax is None:
        return w, f

    lo = -np.inf if wmin is None else wmin
    hi = np.inf if wmax is None else wmax
    m = (w >= lo) & (w <= hi)
    return w[m], f[m]


def main():
    ap = argparse.ArgumentParser(
        description="Extract one model gridpoint spectrum (optionally in a wavelength range) from an exported HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("h5file", help="Path to models_<moca_mgridid>.h5")
    ap.add_argument("--gridpoint-id", type=int, required=True, help="model_gridpoint_id to extract")
    ap.add_argument("--wmin", type=float, default=None, help="Min wavelength (Angstrom)")
    ap.add_argument("--wmax", type=float, default=None, help="Max wavelength (Angstrom)")
    ap.add_argument("--out", default=None, help="Output CSV file. If omitted, prints a short summary.")
    ap.add_argument("--show-meta", action="store_true", help="Print some metadata/params to stdout")
    args = ap.parse_args()

    with h5py.File(args.h5file, "r") as h5:
        mode = h5["spectra"].attrs.get("mode", "unknown")
        # h5py can return bytes for attrs; normalize
        if isinstance(mode, (bytes, bytearray)):
            mode = mode.decode("utf-8", "replace")

        i = _find_index_of_gridpoint(h5, args.gridpoint_id)

        # Optional: show some context
        if args.show_meta:
            mgridid = h5["meta"].attrs.get("moca_mgridid", None)
            fname = h5["gridpoints/file_name"][i]
            if isinstance(fname, (bytes, bytearray)):
                fname = fname.decode("utf-8", "replace")
            print(f"mode={mode}")
            print(f"moca_mgridid={mgridid}")
            print(f"gridpoint_index={i}")
            print(f"file_name={fname}")

            # Parameters
            par_names = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                         for x in h5["parameters/names"][:]]
            vals = h5["gridpoints/param_values"][i, :]
            # Print non-nan only
            for name, val in zip(par_names, vals):
                if np.isfinite(val):
                    print(f"  {name}={val:g}")

        if mode == "common_grid":
            w, f = _load_common_grid(h5, i, args.wmin, args.wmax)
        elif mode == "ragged_concat":
            w, f = _load_ragged(h5, i, args.wmin, args.wmax)
        else:
            raise SystemExit(f"ERROR: Unknown spectra mode '{mode}'. Expected common_grid or ragged_concat.")

    if args.out:
        header = "wavelength_angstrom,flux_flambda\n"
        data = np.column_stack([w, f])
        np.savetxt(args.out, data, delimiter=",", header=header, comments="")
        print(f"Wrote {len(w):,} points -> {args.out}")
    else:
        # Minimal output
        if w.size == 0:
            print("No data for that gridpoint (empty spectrum).")
        else:
            print(f"Loaded {w.size:,} points for gridpoint_id={args.gridpoint_id}")
            print(f"wavelength range: {w.min():.3f} .. {w.max():.3f} Angstrom")
            # Print a tiny preview
            n = min(5, w.size)
            print("preview:")
            for j in range(n):
                print(f"  {w[j]:.6f}  {f[j]:.6e}")


if __name__ == "__main__":
    main()