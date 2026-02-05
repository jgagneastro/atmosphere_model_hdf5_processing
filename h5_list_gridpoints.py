#!/usr/bin/env python3
import argparse
import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser(
        description="Inspect gridpoints in an exported model-grid HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("h5file")
    ap.add_argument("--head", type=int, default=10, help="How many gridpoints to list")
    ap.add_argument("--gridpoint-id", type=int, default=None, help="If set, print that gridpoint's params")
    args = ap.parse_args()

    with h5py.File(args.h5file, "r") as h5:
        gp_ids = h5["gridpoints/model_gridpoint_id"][:]
        fnames = h5["gridpoints/file_name"][:]
        mode = h5["spectra"].attrs.get("mode", "unknown")
        if isinstance(mode, (bytes, bytearray)):
            mode = mode.decode("utf-8", "replace")
        print(f"mode={mode}")
        print(f"ngridpoints={len(gp_ids):,}")

        n = min(args.head, len(gp_ids))
        print(f"\nFirst {n} gridpoints:")
        for i in range(n):
            fn = fnames[i]
            if isinstance(fn, (bytes, bytearray)):
                fn = fn.decode("utf-8", "replace")
            print(f"  idx={i:6d}  model_gridpoint_id={int(gp_ids[i])}  file_name={fn}")

        if args.gridpoint_id is not None:
            hits = np.where(gp_ids == args.gridpoint_id)[0]
            if hits.size == 0:
                raise SystemExit(f"ERROR: gridpoint_id={args.gridpoint_id} not found")
            i = int(hits[0])

            par_names = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                         for x in h5["parameters/names"][:]]
            vals = h5["gridpoints/param_values"][i, :]

            fn = fnames[i]
            if isinstance(fn, (bytes, bytearray)):
                fn = fn.decode("utf-8", "replace")

            print(f"\nGridpoint {args.gridpoint_id} (index {i}) file_name={fn}")
            for name, val in zip(par_names, vals):
                if np.isfinite(val):
                    print(f"  {name}={val:g}")


if __name__ == "__main__":
    main()