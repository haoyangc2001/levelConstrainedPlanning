"""CLI for standalone SR5 level constrained planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _load_request(path: str | Path) -> dict:
    request_path = Path(path)
    text = request_path.read_text(encoding="utf-8")
    if request_path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"request must be a mapping: {request_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m level_planner.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="run one SR5 level-constrained planning request")
    plan.add_argument("--config", required=True, help="planner config YAML")
    plan.add_argument("--request", required=True, help="request JSON/YAML")
    plan.add_argument("--out", required=True, help="output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        from level_planner_core import LevelConstrainedPlanner

        planner = LevelConstrainedPlanner.from_config(args.config)
        result = planner.plan(_load_request(args.request), out_dir=args.out)
        print(json.dumps({
            "request_id": result.get("request_id"),
            "status": result.get("status"),
            "failure_reason": result.get("failure_reason"),
            "success_source": result.get("metrics", {}).get("success_source"),
            "selected_candidate_id": result.get("metrics", {}).get("selected_candidate_id"),
            "result_json": result.get("artifacts", {}).get("result_json"),
        }, ensure_ascii=False))
        return 0 if result.get("status") == "success" else 2
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
