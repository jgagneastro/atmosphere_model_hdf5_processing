#!/usr/bin/env python3
import os
import math
import numpy as np
import h5py
import pymysql

# -----------------------
# DB helpers
# -----------------------
def db_connect():
    # Expect your usual env vars (adapt to your ~/.moca_db.env)
    host = os.environ["MOCA_HOST"]
    user = os.environ["MOCA_USERNAME"]
    passwd = os.environ["MOCA_PASSWORD"]
    db = os.environ["MOCA_DBNAME"]
    port = int(os.environ.get("MOCA_PORT", "3306"))
    return pymysql.connect(
        host=host, user=user, password=passwd, database=db, port=port,
        charset="utf8mb4", autocommit=True
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
def export_one_mgridid(mgridid, out_path, sample_check_n=3, compression="lzf"):
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

        # moca_mgridfileid per gridpoint (should be unique)
        gp_to_fileid = {}
        for r in rows:
            gp = r["model_gridpoint_id"]
            fid = int(r["moca_mgridfileid"])
            prev = gp_to_fileid.get(gp)
            if prev is None:
                gp_to_fileid[gp] = fid
            elif prev != fid:
                raise RuntimeError(f"Gridpoint {gp} has multiple moca_mgridfileid: {prev} vs {fid}")

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
                if v is None:
                    continue
                # HDF5 attrs don't love long TEXT; store short-ish ones safely
                if isinstance(v, (bytes, str)):
                    meta_grp.attrs[k] = str(v)
                else:
                    meta_grp.attrs[k] = v

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--moca_mgridid", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--compression", default="lzf", choices=["lzf", "gzip"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"models_{args.moca_mgridid}.h5")
    export_one_mgridid(args.moca_mgridid, out_path, compression=args.compression)