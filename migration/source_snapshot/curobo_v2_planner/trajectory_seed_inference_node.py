#!/usr/bin/env python3
"""Trajectory seed inference node skeleton for diffusion shadow mode.

[caohy] diffusionSeedLearning phase 5: this node is intentionally conservative.
It exposes only a status service and does not generate runtime candidates yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger


class TrajectorySeedInferenceNode(Node):
    """Optional diffusion seed inference process."""

    def __init__(self) -> None:
        super().__init__("trajectory_seed_inference")
        self.declare_parameter("mode", "off")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("generated_samples_path", "")
        self.declare_parameter("model_timeout_sec", 0.2)
        self.declare_parameter("k_generate", 0)
        self.declare_parameter("k_accept", 0)
        self.declare_parameter("max_start_gap_l2", 0.05)
        self.declare_parameter("max_step_l2", 1.0)
        self.declare_parameter("fallback_to_rule_seed", True)
        self.mode = str(self.get_parameter("mode").value).strip().lower()
        self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        self.generated_samples_path = str(self.get_parameter("generated_samples_path").value)
        self.model_timeout_sec = float(self.get_parameter("model_timeout_sec").value)
        self.k_generate = int(self.get_parameter("k_generate").value)
        self.k_accept = int(self.get_parameter("k_accept").value)
        self.max_start_gap_l2 = float(self.get_parameter("max_start_gap_l2").value)
        self.max_step_l2 = float(self.get_parameter("max_step_l2").value)
        self.fallback_to_rule_seed = self._coerce_bool(
            self.get_parameter("fallback_to_rule_seed").value
        )
        self._status_srv = self.create_service(
            Trigger,
            "trajectory_seed_inference/status",
            self._handle_status,
        )
        self.get_logger().info(
            "trajectory_seed_inference skeleton started: "
            f"mode={self.mode}, checkpoint={self.checkpoint_path or '<none>'}, "
            f"k_generate={self.k_generate}, timeout={self.model_timeout_sec}"
        )

    def _handle_status(self, request, response):  # noqa: ANN001
        checkpoint_exists = bool(self.checkpoint_path and Path(self.checkpoint_path).exists())
        generated_samples_exists = bool(
            self.generated_samples_path and Path(self.generated_samples_path).exists()
        )
        payload = {
            "provider_name": "diffusion_seed",
            "status": "skeleton_ready",
            "mode": self.mode,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_exists": checkpoint_exists,
            "generated_samples_path": self.generated_samples_path,
            "generated_samples_exists": generated_samples_exists,
            "model_timeout_sec": self.model_timeout_sec,
            "k_generate": self.k_generate,
            "k_accept": self.k_accept,
            "max_start_gap_l2": self.max_start_gap_l2,
            "max_step_l2": self.max_step_l2,
            "fallback_to_rule_seed": self.fallback_to_rule_seed,
            "runtime_effect": "status_only_planner_owns_candidate_pool",
        }
        response.success = True
        response.message = json.dumps(payload, ensure_ascii=False)
        return response

    @staticmethod
    def _coerce_bool(value) -> bool:  # noqa: ANN001
        if isinstance(value, bool):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectorySeedInferenceNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
