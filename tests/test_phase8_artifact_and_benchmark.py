from __future__ import annotations

import json
from pathlib import Path

import yaml

from level_planner_core.planner import LevelPlannerConfig
from tools.dataset.run_closed_loop_benchmark import strategy_request


def test_config_loads_model_paths_from_artifact_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / "artifacts.json"
    pointer.write_text(
        json.dumps(
            {
                "diffusion": {
                    "best_checkpoint": "/pub/data/example/diffusion/best.pt",
                },
                "critic": {
                    "best_checkpoint": "/pub/data/example/critic/best.pt",
                },
                "generated_samples": {
                    "path": "/pub/data/example/generated.json",
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "planner.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "planner": {
                    "artifact_pointer": str(pointer),
                    "load_model_paths_from_artifacts": True,
                    "diffusion_checkpoint_path": "fallback_diffusion.pt",
                    "critic_checkpoint_path": "fallback_critic.pt",
                }
            }
        ),
        encoding="utf-8",
    )
    config = LevelPlannerConfig.from_file(config_path)
    assert config.diffusion_checkpoint_path == "/pub/data/example/diffusion/best.pt"
    assert config.critic_checkpoint_path == "/pub/data/example/critic/best.pt"
    assert config.diffusion_generated_samples_path == "/pub/data/example/generated.json"


def test_benchmark_strategy_request_disables_native_for_pure_modes() -> None:
    base = {
        "request_id": "req",
        "seed_policy": {
            "mode": "mixed",
            "fallback_to_rule_seed": True,
        },
        "metadata": {},
    }
    rule = strategy_request(base, "rule_only", 2500.0)
    diffusion = strategy_request(base, "diffusion_only", 2500.0)
    mixed = strategy_request(base, "mixed_fallback", 2500.0)

    assert rule["seed_policy"]["mode"] == "rule"
    assert rule["seed_policy"]["fallback_to_planner_native"] is False
    assert diffusion["seed_policy"]["mode"] == "diffusion"
    assert diffusion["seed_policy"]["fallback_to_rule_seed"] is False
    assert diffusion["seed_policy"]["fallback_to_planner_native"] is False
    assert mixed["seed_policy"]["mode"] == "mixed"
    assert mixed["seed_policy"]["fallback_to_rule_seed"] is True
    assert mixed["seed_policy"]["fallback_to_planner_native"] is True
    assert mixed["metadata"]["total_budget_ms"] == 2500.0
