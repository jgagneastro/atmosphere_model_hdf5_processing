#!/usr/bin/env python3
import os
import math
import numpy as np
import h5py
import pymysql

# -----------------------
# DB helpers
# -----------------------
def _require_env(var_name: str) -> str:
    v = os.environ.get(var_name)
    if v is None or str(v).strip() == "":
        raise RuntimeError(
            f"Missing required environment variable {var_name}. "
            f"Did you `source ~/.moca_db.env` (or equivalent) before running?"
        )
    return v


def db_connect():
    # Expect your usual env vars (adapt to your ~/.moca_db.env)
    host = _require_env("ATM_HOST")
    user = _require_env("ATM_USERNAME")
    passwd = _require_env("ATM_PASSWORD")
    db = _require_env("ATM_DBNAME")
    port = int(os.environ.get("ATM_PORT", "3306"))
    return pymysql.connect(
        host=host,
        user=user,
        password=passwd,
        database=db,
        port=port,
        charset="utf8mb4",
        autocommit=True,
    )

def fetchall_dict(conn, sql, args=()):
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(sql, args)
        return cur.fetchall()

def fetchone_dict(conn, sql, args=()):
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(sql, args)
        return cur.fetchone()

# -----------------------
# HDF5 helpers
# -----------------------
def _to_h5_attr(v):
    """Convert arbitrary DB values to something h5py can store as an attribute."""
    if v is None:
        return None

    # bytes -> utf-8 string (safe fallback)
    if isinstance(v, (bytes, bytearray, memoryview)):
        try:
            return bytes(v).decode("utf-8")
        except Exception:
            return bytes(v).hex()

    # numpy scalars -> python scalars
    if isinstance(v, np.generic):
        return v.item()

    # datetime/date/time objects -> ISO string
    try:
        import datetime as _dt
        if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
            return v.isoformat()
    except Exception:
        pass

    # decimal.Decimal -> float if finite, else string
    try:
        import decimal as _dec
        if isinstance(v, _dec.Decimal):
            try:
                return float(v)
            except Exception:
                return str(v)
    except Exception:
        pass

    # Basic scalar types are OK
    if isinstance(v, (str, int, float, bool)):
        return v

    # Lists/tuples: try numeric array, otherwise stringify
    if isinstance(v, (list, tuple)):
        try:
            arr = np.asarray(v)
            # If it's an object array, h5py attrs will choke; store as string
            if arr.dtype == object:
                return str(v)
            return arr
        except Exception:
            return str(v)

    # Dicts/sets/other objects: store as string
    return str(v)

# -----------------------
# Common-grid check
# -----------------------
def get_wavelengths_for_gridpoint(conn, mgridid, gridpoint_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT wavelength_angstrom
            FROM data_model_grid_spectra
            WHERE moca_mgridid=%s AND model_gridpoint_id=%s AND ignored=0
            ORDER BY wavelength_angstrom
            """,
            (mgridid, gridpoint_id)
        )
        arr = [row[0] for row in cur.fetchall()]
    return np.asarray(arr, dtype=np.float64)

def is_common_grid(conn, mgridid, gridpoint_ids, rtol=0.0, atol=0.0):
    # Compare first vs a few others
    ref = get_wavelengths_for_gridpoint(conn, mgridid, gridpoint_ids[0])
    if ref.size == 0:
        raise RuntimeError(f"No spectra rows for moca_mgridid={mgridid}")
    for gp in gridpoint_ids[1:]:
        w = get_wavelengths_for_gridpoint(conn, mgridid, gp)
        if w.size != ref.size:
            return False, ref
        if not np.allclose(w, ref, rtol=rtol, atol=atol):
            return False, ref
    return True, ref

# -----------------------
# Main export
# -----------------------
def export_one_mgridid(mgridid, out_path, sample_check_n=3, compression="lzf", allow_multiple_fileids: bool = True):
    conn = db_connect()
    try:
        # ---- moca_model_grids metadata
        meta = fetchone_dict(
            conn,
            "SELECT * FROM moca_model_grids WHERE moca_mgridid=%s",
            (mgridid,)
        )
        if not meta:
            raise RuntimeError(f"Unknown moca_mgridid={mgridid}")

        # ---- parameter definitions
        pars = fetchall_dict(
            conn,
            """
            SELECT moca_mgridpar, physical_units, lower_bound, upper_bound
            FROM moca_model_grid_parameters
            WHERE moca_mgridid=%s
            ORDER BY moca_mgridpar
            """,
            (mgridid,)
        )
        par_names = [p["moca_mgridpar"] for p in pars]
        npar = len(par_names)

        # ---- gridpoints (pivot in python)
        rows = fetchall_dict(
            conn,
            """
            SELECT model_gridpoint_id, moca_mgridfileid, moca_mgridpar, parameter_value
            FROM data_model_grid_points
            WHERE moca_mgridid=%s
            ORDER BY model_gridpoint_id, moca_mgridpar
            """,
            (mgridid,)
        )
        if not rows:
            raise RuntimeError(f"No gridpoints in data_model_grid_points for {mgridid}")

        # Unique ordered gridpoint ids
        gp_ids = sorted({r["model_gridpoint_id"] for r in rows})
        ngp = len(gp_ids)
        gp_index = {gp: i for i, gp in enumerate(gp_ids)}

        # moca_mgridfileid per gridpoint (allow multiple if configured)
        # Build mapping: gridpoint -> set of fileids
        gp_to_fileids = {}
        for r in rows:
            gp = r["model_gridpoint_id"]
            fid = int(r["moca_mgridfileid"])
            if gp not in gp_to_fileids:
                gp_to_fileids[gp] = set()
            gp_to_fileids[gp].add(fid)

        conflicts = {gp: sorted(list(fids)) for gp, fids in gp_to_fileids.items() if len(fids) > 1}
        if conflicts and not allow_multiple_fileids:
            # Keep message readable
            sample_items = list(conflicts.items())[:10]
            sample_str = "; ".join([f"gp={gp} fileids={fids}" for gp, fids in sample_items])
            raise RuntimeError(
                f"model_gridpoint_id values mapping to multiple moca_mgridfileid values are not allowed in strict mode. "
                f"Example: {sample_str}.\n"
                f"(Your schema allows this because data_model_grid_points is UNIQUE on (moca_mgridpar, moca_mgridid, model_gridpoint_id) without moca_mgridfileid.)\n"
                f"Re-run without strict mode (default) or pass --allow-multiple-fileids to proceed."
            )

        # For all gridpoints, pick the smallest fileid (for backwards compatibility and reproducibility)
        gp_to_fileid = {gp: min(fids) for gp, fids in gp_to_fileids.items()}

        # Fill param matrix
        param_values = np.full((ngp, npar), np.nan, dtype=np.float32)
        par_index = {name: j for j, name in enumerate(par_names)}
        for r in rows:
            i = gp_index[r["model_gridpoint_id"]]
            j = par_index[r["moca_mgridpar"]]
            param_values[i, j] = float(r["parameter_value"])

        # ---- file names
        files = fetchall_dict(
            conn,
            """
            SELECT moca_mgridfileid, file_name
            FROM data_model_grid_files
            WHERE moca_mgridid=%s AND ignored=0
            """,
            (mgridid,)
        )
        file_name_by_id = {int(f["moca_mgridfileid"]): (f["file_name"] or "") for f in files}
        gp_fileids = np.asarray([gp_to_fileid[gp] for gp in gp_ids], dtype=np.int32)
        gp_filenames = [file_name_by_id.get(int(fid), "") for fid in gp_fileids]

        # ---- quick common-grid test
        check_ids = gp_ids[: min(sample_check_n, len(gp_ids))]
        common, ref_wavelength = is_common_grid(conn, mgridid, check_ids)
        # You can loosen tolerance if needed:
        # common, ref_wavelength = is_common_grid(conn, mgridid, check_ids, atol=1e-12)

        # ---- create HDF5
        with h5py.File(out_path, "w") as h5:
            # meta as attrs
            meta_grp = h5.create_group("meta")
            for k, v in meta.items():
                hv = _to_h5_attr(v)
                if hv is None:
                    continue
                try:
                    # Store short-ish ones safely; for anything else, _to_h5_attr will coerce
                    meta_grp.attrs[k] = hv
                except TypeError:
                    # Last-resort fallback for object dtypes
                    meta_grp.attrs[k] = str(v)

            # parameters
            pgrp = h5.create_group("parameters")
            dt_str = h5py.string_dtype(encoding="utf-8")
            pgrp.create_dataset("names", data=np.asarray(par_names, dtype=object), dtype=dt_str)
            pgrp.create_dataset("units", data=np.asarray([p["physical_units"] or "" for p in pars], dtype=object), dtype=dt_str)
            pgrp.create_dataset("lower_bound", data=np.asarray([p["lower_bound"] if p["lower_bound"] is not None else np.nan for p in pars], dtype=np.float32))
            pgrp.create_dataset("upper_bound", data=np.asarray([p["upper_bound"] if p["upper_bound"] is not None else np.nan for p in pars], dtype=np.float32))

            # gridpoints
            ggrp = h5.create_group("gridpoints")
            ggrp.create_dataset("model_gridpoint_id", data=np.asarray(gp_ids, dtype=np.int32))
            ggrp.create_dataset("moca_mgridfileid", data=gp_fileids)
            ggrp.create_dataset("file_name", data=np.asarray(gp_filenames, dtype=object), dtype=dt_str)
            ggrp.create_dataset("param_values", data=param_values, compression=compression, shuffle=True)

            sgrp = h5.create_group("spectra")

            if common:
                # ---- Mode A: common wavelength axis
                w = ref_wavelength
                nlam = w.size
                sgrp.create_dataset("wavelength", data=w, dtype=np.float64)

                # chunking: tune to your typical access pattern
                # Here: chunk by (some gridpoints) x (some wavelengths)
                chunk_gp = min(256, ngp)
                chunk_lam = min(4096, nlam)
                flux_ds = sgrp.create_dataset(
                    "flux",
                    shape=(ngp, nlam),
                    dtype=np.float32,
                    chunks=(chunk_gp, chunk_lam),
                    compression=compression,
                    shuffle=True,
                    fillvalue=np.nan,
                )

                # Stream spectra in one ordered pass
                # Use server-side cursor to avoid loading everything
                cur = conn.cursor(pymysql.cursors.SSCursor)
                cur.execute(
                    """
                    SELECT model_gridpoint_id, wavelength_angstrom, flux_flambda
                    FROM data_model_grid_spectra
                    WHERE moca_mgridid=%s AND ignored=0
                    ORDER BY model_gridpoint_id, wavelength_angstrom
                    """,
                    (mgridid,)
                )

                current_gp = None
                current_row = None
                current_w = []
                current_f = []

                def flush_one(gp, w_list, f_list):
                    if gp is None:
                        return
                    i = gp_index.get(gp)
                    if i is None:
                        return
                    # Optional strict check: wavelengths match reference
                    # (Comment out if too slow)
                    if len(w_list) != nlam:
                        raise RuntimeError(f"{mgridid}: gp={gp} nlam mismatch {len(w_list)} vs {nlam}")
                    # Write row
                    flux_ds[i, :] = np.asarray(f_list, dtype=np.float32)

                for gp, wav, flx in cur:
                    gp = int(gp)
                    if current_gp is None:
                        current_gp = gp
                    if gp != current_gp:
                        flush_one(current_gp, current_w, current_f)
                        current_gp = gp
                        current_w = []
                        current_f = []
                    current_w.append(float(wav))
                    current_f.append(float(flx) if flx is not None else np.nan)

                flush_one(current_gp, current_w, current_f)
                cur.close()

                sgrp.attrs["mode"] = "common_grid"

            else:
                # ---- Mode B: ragged (CSR-style concat)
                sgrp.attrs["mode"] = "ragged_concat"

                offsets = np.zeros((ngp,), dtype=np.int64)
                lengths = np.zeros((ngp,), dtype=np.int32)

                w_ds = sgrp.create_dataset(
                    "wavelength_concat",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.float64,
                    chunks=(1_000_000,),
                    compression=compression,
                    shuffle=True,
                )
                f_ds = sgrp.create_dataset(
                    "flux_concat",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.float32,
                    chunks=(1_000_000,),
                    compression=compression,
                    shuffle=True,
                )

                cur = conn.cursor(pymysql.cursors.SSCursor)
                cur.execute(
                    """
                    SELECT model_gridpoint_id, wavelength_angstrom, flux_flambda
                    FROM data_model_grid_spectra
                    WHERE moca_mgridid=%s AND ignored=0
                    ORDER BY model_gridpoint_id, wavelength_angstrom
                    """,
                    (mgridid,)
                )

                current_gp = None
                w_buf = []
                f_buf = []
                total = 0

                def append_buf(gp, w_list, f_list, total_count):
                    if gp is None:
                        return total_count
                    i = gp_index.get(int(gp))
                    if i is None:
                        return total_count

                    n = len(w_list)
                    offsets[i] = total_count
                    lengths[i] = n

                    # grow datasets
                    w_ds.resize((total_count + n,))
                    f_ds.resize((total_count + n,))
                    w_ds[total_count: total_count + n] = np.asarray(w_list, dtype=np.float64)
                    f_ds[total_count: total_count + n] = np.asarray(
                        [float(x) if x is not None else np.nan for x in f_list],
                        dtype=np.float32
                    )
                    return total_count + n

                for gp, wav, flx in cur:
                    if current_gp is None:
                        current_gp = gp
                    if gp != current_gp:
                        total = append_buf(current_gp, w_buf, f_buf, total)
                        current_gp = gp
                        w_buf = []
                        f_buf = []
                    w_buf.append(float(wav))
                    f_buf.append(flx)

                total = append_buf(current_gp, w_buf, f_buf, total)
                cur.close()

                sgrp.create_dataset("offsets", data=offsets)
                sgrp.create_dataset("lengths", data=lengths)

        print(f"Wrote: {out_path}")

    finally:
        conn.close()

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Export one MOCA model grid (moca_mgridid) to an HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  source ~/.moca_db.env\n"
            "  ./process_moca_mgridid.py 12 --outdir /scratch/$USER/grids\n"
            "  python process_moca_mgridid.py --moca_mgridid 12 --outdir ./out --compression gzip\n"
        ),
    )

    # Accept mgridid as positional OR as a flag (alias)
    ap.add_argument(
        "moca_mgridid",
        nargs="?",
        help="MOCA model grid id to export (same as --moca_mgridid).",
    )
    ap.add_argument(
        "--moca_mgridid",
        dest="moca_mgridid_flag",
        help="MOCA model grid id to export.",
    )

    ap.add_argument("--outdir", required=True, help="Output directory for the HDF5 file")
    ap.add_argument("--outfile", default=None, help="Optional explicit output filename (overrides default)")
    ap.add_argument("--compression", default="lzf", choices=["lzf", "gzip"], help="HDF5 compression")
    ap.add_argument(
        "--sample-check-n",
        type=int,
        default=3,
        help="How many gridpoints to sample when checking for a common wavelength grid",
    )

    ap.add_argument(
        "--allow-multiple-fileids",
        action="store_true",
        default=True,
        help="(Default: enabled) Allow a model_gridpoint_id to map to multiple moca_mgridfileid values; the exporter chooses the smallest fileid per gridpoint and records conflicts in the HDF5.",
    )
    ap.add_argument(
        "--strict-single-fileid",
        action="store_true",
        help="Disallow multiple moca_mgridfileid per model_gridpoint_id (strict mode).",
    )

    args = ap.parse_args()

    # Resolve moca_mgridid from positional or flag
    mgridid = args.moca_mgridid_flag if args.moca_mgridid_flag is not None else args.moca_mgridid
    if mgridid is None:
        ap.error("moca_mgridid is required (provide positional moca_mgridid or --moca_mgridid)")

    os.makedirs(args.outdir, exist_ok=True)

    if args.outfile is None:
        out_path = os.path.join(args.outdir, f"models_{mgridid}.h5")
    else:
        out_path = args.outfile
        if not os.path.isabs(out_path):
            out_path = os.path.join(args.outdir, out_path)

    export_one_mgridid(
        mgridid,
        out_path,
        sample_check_n=args.sample_check_n,
        compression=args.compression,
        allow_multiple_fileids=(not args.strict_single_fileid),
    )