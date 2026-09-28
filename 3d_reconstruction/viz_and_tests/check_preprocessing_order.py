#!/usr/bin/env python3
"""Check that HDF5 files declare the canonical crop-then-resize geometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


EXPECTED_TRANSFORM = "center_crop_resize"


def as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def collect_transform_labels(path: Path) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    with h5py.File(path, "r") as h5_file:
        if "intrinsics_transform" in h5_file.attrs:
            labels.append(("/", as_text(h5_file.attrs["intrinsics_transform"])))

        def inspect(name: str, item: h5py.Group | h5py.Dataset) -> None:
            if "intrinsics_transform" in item.attrs:
                labels.append((f"/{name}", as_text(item.attrs["intrinsics_transform"])))

        h5_file.visititems(inspect)
    return labels


def resolve_sequence(path: Path) -> Path:
    candidates = [path, Path(__file__).resolve().parents[1] / path]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Sequence directory does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read intrinsics_transform metadata from every HDF5 file in a "
            "sequence and verify the canonical crop-then-resize preprocessing."
        )
    )
    parser.add_argument("--sequence", required=True, type=Path)
    args = parser.parse_args()

    sequence = resolve_sequence(args.sequence)
    h5_paths = sorted({*sequence.rglob("*.h5"), *sequence.rglob("*.hdf5")})
    if not h5_paths:
        raise FileNotFoundError(f"No HDF5 files found in {sequence} or its subdirectories")

    invalid_labels: list[str] = []
    labelled_files = 0
    print(f"Sequence: {sequence}")
    print("HDF5 preprocessing metadata:")

    for path in h5_paths:
        relative_path = path.relative_to(sequence)
        labels = collect_transform_labels(path)
        if not labels:
            print(f"  {relative_path}: no intrinsics_transform label")
            continue
        labelled_files += 1
        descriptions = []
        for location, raw_label in labels:
            valid = raw_label == EXPECTED_TRANSFORM
            if not valid:
                invalid_labels.append(f"{relative_path}:{location}={raw_label!r}")
            descriptions.append(
                f"{location}={raw_label!r} ({'valid' if valid else 'unsupported'})"
            )
        print(f"  {relative_path}: " + "; ".join(descriptions))

    print()
    if invalid_labels:
        print("RESULT: UNSUPPORTED PREPROCESSING METADATA; regenerate these files:")
        for label in invalid_labels:
            print(f"  - {label}")
    elif labelled_files:
        print("RESULT: all labelled files use CENTER CROP THEN RESIZE.")
    else:
        print("RESULT: preprocessing order cannot be determined from HDF5 metadata.")

    print(
        "Note: this checks stored metadata only; it cannot infer the actual "
        "pixel operation when a label is missing or incorrect."
    )


if __name__ == "__main__":
    main()
