#!/usr/bin/env python3
"""Download the Event_and_Depth dataset from the Hugging Face Hub.

The dataset (https://huggingface.co/datasets/mthor/Event_and_Depth) contains
42 training and 6 evaluation sequences in the layout that the training,
evaluation, and reconstruction scripts expect:

    data/Event_and_Depth/{train,eval}/<sequence>/...

By default every file of the train and eval sequences is downloaded (about
154 GB). --precomputed_only restricts the download to the model inputs that
training, evaluation, and reconstruction read (about 60 GB), --raw_only to
the recorder files needed to rerun data_precomputation/precompute_all.sh
(about 91 GB). --eval_only restricts the download to the evaluation
sequences and can be combined with either of the two.
Files that are already complete are skipped, so an interrupted download can
simply be restarted.

Usage:
    python3 download_dataset.py                                # everything
    python3 download_dataset.py --precomputed_only             # model inputs only
    python3 download_dataset.py --raw_only                     # inputs for precompute_all.sh
    python3 download_dataset.py --eval_only --precomputed_only # evaluation inputs only
    python3 download_dataset.py --sequences 20 25              # selected sequences only
    python3 download_dataset.py --dry_run                      # list sizes, download nothing
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

from config import DATA_ROOT

REPO_ID = "mthor/Event_and_Depth"
REPO_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "eval")
FILE_SETS = {
    # Inputs of training, evaluation, and reconstruction.
    "precomputed": (
        "events/voxels_cam0.h5",
        "hdf5/depth_in_event_frame.h5",
        "hdf5/poses.h5",
        "hdf5/table_plane.h5",
    ),
    # Recorder files from which precompute_all.sh derives the inputs above.
    "raw": (
        "raw_event_data/events_cam0.raw",
        "hdf5/realsense.h5",
        "hdf5/poses.h5",
    ),
}


def format_size(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.1f} GB"


def free_space(path: Path) -> int:
    """Free bytes on the file system that holds path or its nearest parent."""
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Download the {REPO_ID} dataset from the Hugging Face Hub."
    )
    parser.add_argument("--out_dir", type=Path, default=REPO_DIR / DATA_ROOT,
                        help="Target directory (default: data/Event_and_Depth)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Download only the evaluation sequences")
    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument("--precomputed_only", action="store_true",
                            help="Download only the inputs of training, evaluation, and reconstruction")
    file_group.add_argument("--raw_only", action="store_true",
                            help="Download only the recorder files that precompute_all.sh needs")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Download only these sequence names, for example 20 25")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel downloads")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print what would be downloaded")
    args = parser.parse_args()
    splits = ("eval",) if args.eval_only else SPLITS
    file_set = "precomputed" if args.precomputed_only else "raw" if args.raw_only else "all"

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        sys.exit(
            "[ERROR] huggingface_hub is not installed. Run "
            "'python3 -m pip install huggingface_hub' or use the Docker image."
        )

    selected = []
    for entry in HfApi().list_repo_tree(REPO_ID, repo_type="dataset", recursive=True):
        parts = entry.path.split("/", 2)
        if getattr(entry, "size", None) is None or len(parts) != 3:
            continue  # folders and repository files such as .gitattributes
        split, sequence, relative = parts
        if split not in splits or relative.endswith(".tmp_index"):
            continue  # .tmp_index files are caches that Metavision rebuilds
        if args.sequences is not None and sequence not in args.sequences:
            continue
        if file_set != "all" and relative not in FILE_SETS[file_set]:
            continue
        selected.append(entry)
    if not selected:
        sys.exit("[ERROR] No dataset files match the selected splits and sequences.")
    if args.sequences is not None:
        found = {entry.path.split("/")[1] for entry in selected}
        unknown = sorted(set(args.sequences) - found)
        if unknown:
            sys.exit(f"[ERROR] Sequences not found in {', '.join(splits)}: {unknown}")

    missing = [
        entry for entry in selected
        if not (
            (args.out_dir / entry.path).is_file()
            and (args.out_dir / entry.path).stat().st_size == entry.size
        )
    ]
    sequences = Counter(entry.path.split("/")[0] for entry in selected if entry.path.endswith("poses.h5"))
    print(f"Dataset   : https://huggingface.co/datasets/{REPO_ID}")
    print(f"Target    : {args.out_dir}")
    print(f"Sequences : {', '.join(f'{count} {split}' for split, count in sequences.items())}")
    print(f"Files     : {len(selected)} ({file_set}), {format_size(sum(e.size for e in selected))}")
    needed = sum(entry.size for entry in missing)
    available = free_space(args.out_dir)
    print(f"To fetch  : {len(missing)} files, {format_size(needed)} "
          f"({format_size(available)} free)")
    if args.dry_run or not missing:
        return
    if needed > available:
        sys.exit("[ERROR] Not enough free disk space for the selected files.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.out_dir,
        allow_patterns=[entry.path for entry in selected],
        max_workers=args.workers,
    )
    print(f"\nDone. Dataset stored in {args.out_dir}")
    if file_set == "raw":
        print("Run ./data_precomputation/precompute_all.sh to create the model inputs.")


if __name__ == "__main__":
    main()
