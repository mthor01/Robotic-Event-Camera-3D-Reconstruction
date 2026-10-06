#!/usr/bin/env python3
"""Extract raw-recorder and minimal precomputation-source datasets.

The source is expected to contain sequence directories, normally below
``train/`` and ``eval/``. Two independent copies are produced:

* ``dataset`` contains only files written by the recorder.
* ``core_dataset`` contains only the recorder inputs required to run the full
  precomputation pipeline.

Precomputation products are selected explicitly for the core dataset and are
never copied into the raw recorder dataset.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_REQUIRED = (
    Path("raw_event_data/events_cam0.raw"),
    Path("hdf5/realsense.h5"),
    Path("hdf5/events_cam0.h5"),
    Path("hdf5/poses.h5"),
)

RAW_OPTIONAL = (
    Path("hdf5/raw_poses.h5"),
    Path("hdf5/metadata.h5"),
    Path("videos/realsense_depth.mp4"),
    Path("videos/realsense_rgb.mp4"),
    Path("videos/events_cam0.mp4"),
)

CORE_REQUIRED = (
    Path("raw_event_data/events_cam0.raw"),
    Path("hdf5/realsense.h5"),
    Path("hdf5/poses.h5"),
)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def discover_sequences(source: Path) -> list[Path]:
    """Find sequence directories from their recorder-created RealSense file."""
    if (source / "hdf5" / "realsense.h5").is_file():
        return [source]
    return sorted(
        file.parent.parent
        for file in source.rglob("hdf5/realsense.h5")
        if file.is_file()
    )


def validate_output(source: Path, output: Path, label: str) -> None:
    if output == source:
        raise ValueError(f"{label} output must differ from the source: {output}")
    if source in output.parents:
        raise ValueError(f"{label} output must not be inside the source: {output}")
    if output in source.parents:
        raise ValueError(f"{label} output must not contain the source: {output}")
    if output == PROJECT_DIR or output == PROJECT_DIR.parent or output == Path("/"):
        raise ValueError(f"Refusing unsafe {label} output: {output}")


def prepare_output(output: Path, overwrite: bool, dry_run: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output}\n"
                "Choose another path or pass --overwrite to replace it."
            )
        if not dry_run:
            shutil.rmtree(output)
    if not dry_run:
        output.mkdir(parents=True, exist_ok=False)


def validate_manifest(
    sequences: list[Path],
    source: Path,
    manifest: tuple[Path, ...],
    label: str,
) -> None:
    missing = [
        f"{sequence.relative_to(source)}/{relative}"
        for sequence in sequences
        for relative in manifest
        if not (sequence / relative).is_file()
    ]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"Cannot create {label}; {len(missing)} required file(s) are missing:\n"
            f"{preview}{suffix}"
        )


def copy_manifest(
    sequences: list[Path],
    source: Path,
    output: Path,
    manifest: tuple[Path, ...],
    dry_run: bool,
) -> tuple[int, int]:
    file_count = 0
    byte_count = 0
    for sequence in sequences:
        sequence_relative = sequence.relative_to(source)
        for relative in manifest:
            source_file = sequence / relative
            if not source_file.is_file():
                continue
            destination = output / sequence_relative / relative
            size = source_file.stat().st_size
            print(f"  {source_file} -> {destination}")
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination)
            file_count += 1
            byte_count += size
    return file_count, byte_count


def format_size(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create raw-recorder and minimal precomputation-source datasets."
    )
    parser.add_argument(
        "--source",
        default="data/new_2",
        help="Processed source dataset (default: data/new_2).",
    )
    parser.add_argument(
        "--dataset-output",
        default="data/dataset",
        help="Raw recorder-only output (default: data/dataset).",
    )
    parser.add_argument(
        "--core-output",
        default="data/core_dataset",
        help="Minimal precomputation-source output (default: data/core_dataset).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and replace existing output directories.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Create only core_dataset and leave dataset untouched.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list files without creating outputs.",
    )
    args = parser.parse_args()

    source = resolve_project_path(args.source)
    dataset_output = resolve_project_path(args.dataset_output)
    core_output = resolve_project_path(args.core_output)

    if not source.is_dir():
        parser.error(f"Source dataset does not exist: {source}")
    if not args.core_only and dataset_output == core_output:
        parser.error("--dataset-output and --core-output must differ")
    if (
        not args.core_only
        and (
            dataset_output in core_output.parents
            or core_output in dataset_output.parents
        )
    ):
        parser.error("Dataset outputs must not contain one another")
    try:
        if not args.core_only:
            validate_output(source, dataset_output, "dataset")
        validate_output(source, core_output, "core dataset")
    except ValueError as error:
        parser.error(str(error))

    sequences = discover_sequences(source)
    if not sequences:
        parser.error(f"No sequences containing hdf5/realsense.h5 found below {source}")

    # Validate everything before creating or deleting either output.
    if not args.core_only:
        validate_manifest(sequences, source, RAW_REQUIRED, "recorder dataset")
    validate_manifest(sequences, source, CORE_REQUIRED, "core dataset")

    if not args.overwrite:
        requested_outputs = (
            (core_output,) if args.core_only else (dataset_output, core_output)
        )
        existing_outputs = [output for output in requested_outputs if output.exists()]
        if existing_outputs:
            formatted = "\n".join(f"  - {output}" for output in existing_outputs)
            parser.error(
                "Output directories already exist:\n"
                f"{formatted}\nChoose other paths or pass --overwrite to replace them."
            )

    print(f"Source:       {source}")
    print(f"Sequences:    {len(sequences)}")
    print(f"Dataset:      {'skipped' if args.core_only else dataset_output}")
    print(f"Core dataset: {core_output}")
    print(f"Mode:         {'dry run' if args.dry_run else 'copy'}")

    if not args.core_only:
        prepare_output(dataset_output, args.overwrite, args.dry_run)
    prepare_output(core_output, args.overwrite, args.dry_run)

    if not args.core_only:
        print("\nRecorder dataset files:")
        raw_files, raw_bytes = copy_manifest(
            sequences,
            source,
            dataset_output,
            RAW_REQUIRED + RAW_OPTIONAL,
            args.dry_run,
        )
    print("\nCore dataset files:")
    core_files, core_bytes = copy_manifest(
        sequences,
        source,
        core_output,
        CORE_REQUIRED,
        args.dry_run,
    )

    print("\nExtraction complete." if not args.dry_run else "\nDry run complete.")
    if not args.core_only:
        missing_optional = sum(
            not (sequence / relative).is_file()
            for sequence in sequences
            for relative in RAW_OPTIONAL
        )
        print(f"  dataset:      {raw_files} files, {format_size(raw_bytes)}")
        if missing_optional:
            print(f"  optional recorder files absent: {missing_optional}")
    print(f"  core_dataset: {core_files} files, {format_size(core_bytes)}")


if __name__ == "__main__":
    main()
