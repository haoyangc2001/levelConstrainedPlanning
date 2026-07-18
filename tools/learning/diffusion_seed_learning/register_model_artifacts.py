#!/usr/bin/env python3
"""Register trained standalone model artifacts in artifacts/current_artifacts.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _sha256(path: Path | None) -> str | None:
    if not path or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path: Path | None, *, hash_file: bool = True) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "size_bytes": None, "sha256": None}
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "sha256": _sha256(path) if hash_file and exists and path.is_file() else None,
    }


def _read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _model_section(checkpoint_dir: Path, metadata: dict[str, Any], kind: str) -> dict[str, Any]:
    best = checkpoint_dir / "best.pt"
    last = checkpoint_dir / "last.pt"
    section = {
        "run_name": metadata.get("run_name") or checkpoint_dir.name,
        "checkpoint_dir": str(checkpoint_dir),
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
        "metadata": str(checkpoint_dir / "metadata.json"),
        "epochs": metadata.get("epochs"),
        "batch_size": metadata.get("batch_size") or metadata.get("hyperparameters", {}).get("batch_size"),
        "hidden_dim": metadata.get("hidden_dim") or metadata.get("model_config", {}).get("hidden_dim"),
        "horizon": metadata.get("horizon"),
        "sample_count": metadata.get("sample_count"),
        "best_loss": metadata.get("best_loss"),
        "samples_path": metadata.get("samples_path"),
        "samples_sha256": metadata.get("samples_sha256"),
        "model_config": metadata.get("model_config") or {},
        "hyperparameters": metadata.get("hyperparameters") or {},
        "best_checkpoint_file": _file_info(best, hash_file=False),
        "metadata_file": _file_info(checkpoint_dir / "metadata.json"),
    }
    if kind == "diffusion":
        section["diffusion_steps"] = metadata.get("diffusion_steps") or metadata.get("diffusion_config", {}).get("steps")
        section["diffusion_config"] = metadata.get("diffusion_config") or {}
        # Record which auxiliary loss components were trained (L_level etc.), so the
        # artifact pointer unambiguously identifies the "all-components" main model vs
        # C5 ablation variants where a component is turned off.
        section["aux_loss_config"] = metadata.get("aux_loss_config") or {}
    else:
        section["positive_count"] = metadata.get("positive_count")
        section["negative_count"] = metadata.get("negative_count")
        section["source_type_counts"] = metadata.get("source_type_counts") or {}
    return section


def update_pointer(
    *,
    pointer_path: Path,
    diffusion_dir: Path,
    critic_dir: Path,
    generated_samples: Path | None,
    offline_generation_report: Path | None,
    benchmark_report: Path | None,
    benchmark_summary: Path | None,
) -> dict[str, Any]:
    pointer = _read_json(pointer_path)
    if pointer.get("diffusion") and "legacy_phase10_diffusion" not in pointer:
        pointer["legacy_phase10_diffusion"] = pointer.get("diffusion")
    if pointer.get("critic") and "legacy_phase10_critic" not in pointer:
        pointer["legacy_phase10_critic"] = pointer.get("critic")
    diffusion_metadata = _read_json(diffusion_dir / "metadata.json")
    critic_metadata = _read_json(critic_dir / "metadata.json")
    pointer["status"] = "active_standalone_closed_loop_models"
    pointer["updated_at"] = _utc_now()
    pointer["git_commit"] = _git_commit()
    pointer["diffusion"] = _model_section(diffusion_dir, diffusion_metadata, "diffusion")
    pointer["critic"] = _model_section(critic_dir, critic_metadata, "critic")
    if generated_samples:
        pointer["generated_samples"] = _file_info(generated_samples, hash_file=False)
    if offline_generation_report:
        pointer["offline_generation_report"] = _file_info(offline_generation_report)
    pointer["closed_loop_benchmark"] = {
        "report": _file_info(benchmark_report),
        "summary": _file_info(benchmark_summary),
    }
    pointer["runtime_resolution"] = {
        "config": "configs/sr5_level.yaml",
        "artifact_pointer": str(pointer_path),
        "note": "LevelPlannerConfig loads diffusion/critic best_checkpoint from this pointer when load_model_paths_from_artifacts=true.",
    }
    pointer_path.write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pointer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer", type=Path, default=REPO_ROOT / "artifacts/current_artifacts.json")
    parser.add_argument("--diffusion-dir", type=Path, required=True)
    parser.add_argument("--critic-dir", type=Path, required=True)
    parser.add_argument("--generated-samples", type=Path)
    parser.add_argument("--offline-generation-report", type=Path)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--benchmark-summary", type=Path)
    args = parser.parse_args(argv)
    pointer = update_pointer(
        pointer_path=args.pointer,
        diffusion_dir=args.diffusion_dir,
        critic_dir=args.critic_dir,
        generated_samples=args.generated_samples,
        offline_generation_report=args.offline_generation_report,
        benchmark_report=args.benchmark_report,
        benchmark_summary=args.benchmark_summary,
    )
    print(json.dumps(pointer, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
