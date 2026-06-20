from __future__ import annotations

from pathlib import Path


DEFAULT_TB_ROOT = Path(__file__).resolve().parent / "checkpoints" / "tensorboard"


def tensorboard_run_dir(model_name: str, run_name: str, tb_root: Path | str | None = None) -> Path:
    root = Path(tb_root) if tb_root is not None else DEFAULT_TB_ROOT
    return root / model_name / run_name
