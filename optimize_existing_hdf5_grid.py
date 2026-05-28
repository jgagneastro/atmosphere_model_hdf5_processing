#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _dataset_kwargs(src, chunks, compression: str | None):
    kwargs = {}
    if chunks is not None:
        kwargs["chunks"] = chunks
    if compression is not None:
        kwargs["compression"] = compression
        if src.dtype.kind in {"f", "i", "u"}:
            kwargs["shuffle"] = True
    fillvalue = getattr(src, "fillvalue", None)
    if fillvalue is not None:
        try:
            if src.dtype.kind not in {"S", "O", "U"}:
                kwargs["fillvalue"] = fillvalue
        except Exception:
            pass
    return kwargs


def _copy_dataset(src, dst_parent, name: str, relpath: str, args) -> None:
    shape = src.shape
    chunks = src.chunks
    compression = src.compression

    if relpath == "spectra/flux" and len(shape) == 2:
        chunks = (
            max(1, min(int(args.chunk_gridpoints), int(shape[0]))),
            max(1, min(int(args.chunk_wavelengths), int(shape[1]))),
        )
        compression = args.compression
    elif relpath in {"spectra/wavelength_concat", "spectra/flux_concat"} and len(shape) == 1:
        chunks = (max(1, min(int(args.ragged_chunk_length), int(shape[0]))),)
        compression = args.compression

    kwargs = _dataset_kwargs(src, chunks, compression)
    dst = dst_parent.create_dataset(name, shape=shape, dtype=src.dtype, **kwargs)
    _copy_attrs(src, dst)

    if relpath == "spectra/flux" and len(shape) == 2:
        dst.attrs["chunk_gridpoints"] = int(chunks[0])
        dst.attrs["chunk_wavelengths"] = int(chunks[1])
        dst.attrs["chunking_strategy"] = "row_range_optimized"
        row_block = max(1, int(args.row_block))
        for start in range(0, shape[0], row_block):
            stop = min(shape[0], start + row_block)
            dst[start:stop, :] = src[start:stop, :]
            if start == 0 or stop == shape[0] or stop % 1000 == 0:
                _log(f"Copied {relpath} rows {stop:,}/{shape[0]:,}")
        return

    if relpath in {"spectra/wavelength_concat", "spectra/flux_concat"} and len(shape) == 1:
        dst.attrs["chunk_length"] = int(chunks[0])
        dst.attrs["chunking_strategy"] = "ragged_row_optimized"
        block = max(1, int(args.concat_block))
        for start in range(0, shape[0], block):
            stop = min(shape[0], start + block)
            dst[start:stop] = src[start:stop]
            if start == 0 or stop == shape[0] or stop % 5_000_000 == 0:
                _log(f"Copied {relpath} values {stop:,}/{shape[0]:,}")
        return

    data = src[()]
    dst[...] = data


def _copy_item(src, dst_parent, name: str, relpath: str, args) -> None:
    if isinstance(src, h5py.Group):
        dst = dst_parent.create_group(name)
        _copy_attrs(src, dst)
        for child_name, child in src.items():
            child_relpath = f"{relpath}/{child_name}" if relpath else child_name
            _copy_item(child, dst, child_name, child_relpath, args)
        if relpath == "spectra":
            mode = dst.attrs.get("mode", "")
            if isinstance(mode, bytes):
                mode = mode.decode("utf-8")
            if mode == "common_grid":
                dst.attrs["chunking_strategy"] = "row_range_optimized"
            elif mode == "ragged_concat":
                dst.attrs["chunking_strategy"] = "ragged_row_optimized"
        return

    if isinstance(src, h5py.Dataset):
        _copy_dataset(src, dst_parent, name, relpath, args)
        return

    raise TypeError(f"Unsupported HDF5 item at {relpath}: {type(src)!r}")


def optimize_hdf5(input_path: Path, output_path: Path, args) -> None:
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists; pass --overwrite to replace: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Optimizing {input_path} -> {output_path}")
    with h5py.File(input_path, "r") as src_h5, h5py.File(output_path, "w") as dst_h5:
        _copy_attrs(src_h5, dst_h5)
        dst_h5.attrs["optimized_by"] = "optimize_existing_hdf5_grid.py"
        dst_h5.attrs["optimized_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        for name, item in src_h5.items():
            _copy_item(item, dst_h5, name, name, args)
    _log(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite an existing RVBAM/MOCA HDF5 model grid with row/range-optimized chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_h5", type=Path)
    parser.add_argument("output_h5", type=Path)
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip"])
    parser.add_argument("--chunk-gridpoints", type=int, default=32)
    parser.add_argument("--chunk-wavelengths", type=int, default=512)
    parser.add_argument("--ragged-chunk-length", type=int, default=65536)
    parser.add_argument("--row-block", type=int, default=64)
    parser.add_argument("--concat-block", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    optimize_hdf5(args.input_h5, args.output_h5, args)


if __name__ == "__main__":
    main()
