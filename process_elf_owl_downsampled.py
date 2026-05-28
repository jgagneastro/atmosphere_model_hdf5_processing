#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
import h5py
import numpy as np
from scipy.io import readsav


_SOURCE_RE = re.compile(
    r"^spectra_logzz_(?P<log_kzz>[-+0-9.]+)"
    r"_teff_(?P<teff>[-+0-9.]+)"
    r"_grav_(?P<grav>[-+0-9.]+)"
    r"_mh_(?P<m_h>[-+0-9.]+)"
    r"_co_(?P<c_o>[-+0-9.]+)"
    r"_downsampled\.sav$"
)


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _decode(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_dataset(src_group, dst_group, name: str, **kwargs) -> None:
    if name not in src_group:
        return
    data = src_group[name][:]
    if data.dtype.kind in {"S", "O", "U"}:
        dt_str = h5py.string_dtype(encoding="utf-8")
        values = np.asarray([_decode(v) for v in data], dtype=object)
        dst_group.create_dataset(name, data=values, dtype=dt_str, **kwargs)
    else:
        dst_group.create_dataset(name, data=data, **kwargs)


def _common_grid_flux_chunks(
    ngp: int,
    nlam: int,
    chunk_gridpoints: int,
    chunk_wavelengths: int,
) -> tuple[int, int]:
    return (
        max(1, min(int(chunk_gridpoints), int(ngp))),
        max(1, min(int(chunk_wavelengths), int(nlam))),
    )


def _parse_source_params(path: Path) -> dict[str, float]:
    match = _SOURCE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Cannot parse elf/owl source filename: {path.name}")
    raw = {key: float(value) for key, value in match.groupdict().items()}
    grav = raw.pop("grav")
    raw["logg"] = float(np.log10(grav * 100.0))
    return raw


def _param_key(values, ndigits: int = 5) -> tuple[float, ...]:
    return tuple(round(float(value), ndigits) for value in values)


def _source_key(names: list[str], params: dict[str, float], ndigits: int = 5) -> tuple[float, ...]:
    return tuple(round(float(params[name]), ndigits) for name in names)


def _source_name_from_template_name(template_name: str) -> str:
    path = Path(template_name)
    if path.suffix == ".sav" and path.name.endswith("_downsampled.sav"):
        return path.name
    stem = path.stem if path.suffix else path.name
    if stem.endswith("_downsampled"):
        return f"{stem}.sav"
    return f"{stem}_downsampled.sav"


def _load_sav_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = readsav(str(path), python_dict=True, verbose=False)
    wavelength_um = np.asarray(data["wavelength_microns"], dtype=np.float64).ravel()
    flux = np.asarray(data["flux_flambda"], dtype=np.float32).ravel()
    order = np.argsort(wavelength_um)
    wavelength_a = wavelength_um[order] * 1.0e4
    return wavelength_a, flux[order]


def export_elf_owl_downsampled(
    source_dir: Path,
    template_h5: Path,
    out_path: Path,
    compression: str = "lzf",
    chunk_gridpoints: int = 32,
    chunk_wavelengths: int = 512,
    wavelength_atol_angstrom: float = 1.0e-6,
    max_gridpoints: int | None = None,
) -> None:
    source_dir = Path(source_dir)
    template_h5 = Path(template_h5)
    out_path = Path(out_path)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not template_h5.exists():
        raise FileNotFoundError(f"Template HDF5 file not found: {template_h5}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_paths = sorted(source_dir.glob("*.sav"))
    if not source_paths:
        raise RuntimeError(f"No .sav source files found in {source_dir}")

    _log(f"Indexing {len(source_paths):,} elf/owl source files from {source_dir}")
    source_by_name = {path.name: path for path in source_paths}
    source_by_key: dict[tuple[float, ...], Path] = {}

    with h5py.File(template_h5, "r") as template:
        param_names = [_decode(value) for value in template["parameters/names"][:]]
        for path in source_paths:
            params = _parse_source_params(path)
            source_by_key[_source_key(param_names, params)] = path

        gridpoint_ids = np.asarray(template["gridpoints/model_gridpoint_id"][:], dtype=np.int32)
        fileids = np.asarray(template["gridpoints/moca_mgridfileid"][:], dtype=np.int32)
        param_values = np.asarray(template["gridpoints/param_values"][:], dtype=np.float32)
        if "gridpoints/file_name" in template:
            template_file_names = [_decode(value) for value in template["gridpoints/file_name"][:]]
        else:
            template_file_names = [""] * len(gridpoint_ids)

        if max_gridpoints is not None:
            n = int(max_gridpoints)
            gridpoint_ids = gridpoint_ids[:n]
            fileids = fileids[:n]
            param_values = param_values[:n, :]
            template_file_names = template_file_names[:n]

        row_sources: list[Path] = []
        row_source_names: list[str] = []
        for row_index, (values, template_name) in enumerate(zip(param_values, template_file_names)):
            source_path = None
            if template_name:
                source_path = source_by_name.get(_source_name_from_template_name(template_name))
            if source_path is None:
                source_path = source_by_key.get(_param_key(values))
            if source_path is None:
                raise RuntimeError(
                    f"No source .sav file matched template row {row_index} "
                    f"with file_name={template_name!r} and params={values.tolist()}"
                )
            row_sources.append(source_path)
            row_source_names.append(source_path.name)

        first_wavelength, first_flux = _load_sav_spectrum(row_sources[0])
        ngrid, nlam = len(row_sources), len(first_wavelength)
        if first_flux.size != nlam:
            raise RuntimeError(f"First source file has {first_flux.size} fluxes but {nlam} wavelengths")

        chunk_gp, chunk_lam = _common_grid_flux_chunks(
            ngrid,
            nlam,
            chunk_gridpoints=chunk_gridpoints,
            chunk_wavelengths=chunk_wavelengths,
        )
        _log(
            "Writing "
            f"{ngrid:,} gridpoints x {nlam:,} wavelengths to {out_path} "
            f"with chunks=({chunk_gp}, {chunk_lam})"
        )

        dt_str = h5py.string_dtype(encoding="utf-8")
        with h5py.File(out_path, "w") as h5:
            meta_out = h5.create_group("meta")
            if "meta" in template:
                _copy_attrs(template["meta"], meta_out)
            meta_out.attrs["generated_by"] = "process_elf_owl_downsampled.py"
            meta_out.attrs["source_dir"] = str(source_dir)
            meta_out.attrs["source_format"] = "IDL .sav with wavelength_microns and flux_flambda"
            meta_out.attrs["template_h5"] = str(template_h5)
            meta_out.attrs["source_wavelength_unit"] = "micron"
            meta_out.attrs["stored_wavelength_unit"] = "angstrom"

            params_out = h5.create_group("parameters")
            for name in ("names", "units", "lower_bound", "upper_bound"):
                _copy_dataset(template["parameters"], params_out, name)

            grid_out = h5.create_group("gridpoints")
            grid_out.create_dataset("model_gridpoint_id", data=gridpoint_ids, dtype=np.int32)
            grid_out.create_dataset("moca_mgridfileid", data=fileids, dtype=np.int32)
            grid_out.create_dataset(
                "file_name",
                data=np.asarray(template_file_names, dtype=object),
                dtype=dt_str,
            )
            grid_out.create_dataset(
                "source_file_name",
                data=np.asarray(row_source_names, dtype=object),
                dtype=dt_str,
            )
            grid_out.create_dataset("param_values", data=param_values, compression=compression, shuffle=True)

            spec_out = h5.create_group("spectra")
            spec_out.attrs["mode"] = "common_grid"
            spec_out.attrs["source"] = "elf_owl_downsampled_sav"
            spec_out.attrs["chunking_strategy"] = "row_range_optimized"
            spec_out.attrs["wavelength_min_angstrom"] = float(first_wavelength[0])
            spec_out.attrs["wavelength_max_angstrom"] = float(first_wavelength[-1])
            spec_out.attrs["common_grid_reference_source"] = row_sources[0].name
            spec_out.attrs["common_grid_wavelength_atol_angstrom"] = float(wavelength_atol_angstrom)
            spec_out.create_dataset("wavelength", data=first_wavelength, dtype=np.float64)
            flux_ds = spec_out.create_dataset(
                "flux",
                shape=(ngrid, nlam),
                dtype=np.float32,
                chunks=(chunk_gp, chunk_lam),
                compression=compression,
                shuffle=True,
                fillvalue=np.nan,
            )
            flux_ds.attrs["chunk_gridpoints"] = chunk_gp
            flux_ds.attrs["chunk_wavelengths"] = chunk_lam
            flux_ds.attrs["chunking_strategy"] = "row_range_optimized"

            resampled_count = 0
            nan_after_resample_count = 0
            for row_index, source_path in enumerate(row_sources):
                wavelength, flux = _load_sav_spectrum(source_path)
                if wavelength.size != nlam or not np.allclose(
                    wavelength,
                    first_wavelength,
                    rtol=0.0,
                    atol=float(wavelength_atol_angstrom),
                ):
                    flux = np.interp(
                        first_wavelength,
                        wavelength,
                        flux,
                        left=np.nan,
                        right=np.nan,
                    ).astype(np.float32)
                    resampled_count += 1
                    nan_after_resample_count += int(np.count_nonzero(~np.isfinite(flux)))
                flux_ds[row_index, :] = flux.astype(np.float32, copy=False)
                if row_index == 0 or (row_index + 1) % 1000 == 0 or row_index + 1 == ngrid:
                    _log(f"Wrote spectra for {row_index + 1:,}/{ngrid:,} gridpoints")
            spec_out.attrs["resampled_to_common_grid"] = bool(resampled_count)
            spec_out.attrs["resampled_spectra_count"] = int(resampled_count)
            spec_out.attrs["resampled_nan_count"] = int(nan_after_resample_count)
            _log(
                f"Resampled {resampled_count:,}/{ngrid:,} spectra onto the common grid "
                f"(NaN values introduced: {nan_after_resample_count:,})"
            )

    _log(f"Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a row/range-optimized HDF5 file from downsampled Sonora Elf/Owl IDL .sav spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source_dir", type=Path, help="Directory containing *_downsampled.sav files.")
    parser.add_argument("--template-h5", type=Path, required=True, help="Existing MOCA-style HDF5 file to copy grid metadata from.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--outfile", default="models_sonora_elf_owl.h5", help="Output HDF5 filename.")
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip"], help="HDF5 compression.")
    parser.add_argument("--chunk-gridpoints", type=int, default=32, help="Flux chunk size along gridpoint axis.")
    parser.add_argument("--chunk-wavelengths", type=int, default=512, help="Flux chunk size along wavelength axis.")
    parser.add_argument(
        "--wavelength-atol-angstrom",
        type=float,
        default=1.0e-6,
        help="Absolute wavelength tolerance for treating source grids as identical before resampling.",
    )
    parser.add_argument(
        "--max-gridpoints",
        type=int,
        default=None,
        help="Debug option: write only the first N template gridpoints.",
    )
    args = parser.parse_args()

    out_path = Path(args.outfile)
    if not out_path.is_absolute():
        out_path = args.outdir / out_path

    export_elf_owl_downsampled(
        source_dir=args.source_dir,
        template_h5=args.template_h5,
        out_path=out_path,
        compression=args.compression,
        chunk_gridpoints=args.chunk_gridpoints,
        chunk_wavelengths=args.chunk_wavelengths,
        wavelength_atol_angstrom=args.wavelength_atol_angstrom,
        max_gridpoints=args.max_gridpoints,
    )


if __name__ == "__main__":
    main()
