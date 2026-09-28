"""Strict JSON/JSONL and lossless non-pickle NPZ replay."""

import hashlib
from pathlib import Path

import numpy as np

from ._common import canonical_json, loads
from .experiment import ExperimentManifest, validate_run
from .interaction import InteractionState


def write_frames(path, frames):
    text = "".join(canonical_json(frame.to_dict()) + "\n" for frame in frames)
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def read_frames(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield InteractionState.from_dict(loads(line))


def write_npz(path, frames):
    rows = [canonical_json(frame.to_dict()) for frame in frames]
    # This is a lossless replay container, not a dense training tensor.
    np.savez_compressed(path, frames_json=np.asarray(rows, dtype=np.str_))


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        rows = archive["frames_json"]
        if rows.ndim != 1 or rows.dtype.kind != "U":
            raise ValueError("Expected a one-dimensional Unicode JSON array")
        return [InteractionState.from_dict(loads(row)) for row in rows.tolist()]


def load_run(directory):
    root = Path(directory).resolve()
    manifest = ExperimentManifest.from_dict(loads((root / "manifest.json").read_text(encoding="utf-8")))
    frame_path = None
    for artifact in manifest.artifacts:
        path = (root / artifact["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact resolves outside its run directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {artifact['path']}")
        if artifact["role"] == "interaction_frames":
            frame_path = path
    frames = list(read_frames(frame_path))
    validate_run(manifest, frames)
    return manifest, frames

