"""Optional ROS 2 adapter for the standalone level planner core."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _load_request(path: str | Path) -> dict[str, Any]:
    request_path = Path(path)
    text = request_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if request_path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"request must be a mapping: {request_path}")
    return payload


def _apply_seed_mode_override(request: dict[str, Any], mode: str | None) -> dict[str, Any]:
    if not mode:
        return request
    patched = json.loads(json.dumps(request))
    patched.setdefault("seed_policy", {})["mode"] = str(mode)
    return patched


def _run_check() -> int:
    import rclpy  # noqa: F401
    from std_srvs.srv import Trigger  # noqa: F401

    print("level_planner_ros check ok")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="import-only adapter smoke check")
    return parser


class LevelPlannerRosNode:
    """Thin ROS wrapper: Trigger service -> core request dict -> result files."""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from std_srvs.srv import Trigger

        class _Node(Node):
            pass

        self._rclpy = rclpy
        self._trigger_type = Trigger
        self.node = _Node("level_planner_ros")
        self.node.declare_parameter("config", "configs/sr5_level.yaml")
        self.node.declare_parameter("request", "examples/requests/request_level_001.json")
        self.node.declare_parameter("out_dir", "runs/ros_adapter")
        self.node.declare_parameter("diffusion_seed_mode", "")
        self.node.declare_parameter("load_planner_on_start", True)

        self.config_path = str(self.node.get_parameter("config").value)
        self.request_path = str(self.node.get_parameter("request").value)
        self.out_dir = str(self.node.get_parameter("out_dir").value)
        self.diffusion_seed_mode = str(self.node.get_parameter("diffusion_seed_mode").value or "")
        self._planner = None
        if bool(self.node.get_parameter("load_planner_on_start").value):
            self._planner = self._make_planner()

        self._service = self.node.create_service(
            Trigger,
            "plan_default",
            self._handle_plan_default,
        )
        self.node.get_logger().info(
            "level_planner_ros ready: config=%s request=%s out_dir=%s"
            % (self.config_path, self.request_path, self.out_dir)
        )

    def _make_planner(self):
        from level_planner_core import LevelConstrainedPlanner

        return LevelConstrainedPlanner.from_config(self.config_path)

    def _handle_plan_default(self, _request, response):
        try:
            planner = self._planner or self._make_planner()
            request_dict = _apply_seed_mode_override(
                _load_request(self.request_path),
                self.diffusion_seed_mode,
            )
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = Path(self.out_dir) / stamp
            result = planner.plan(request_dict, out_dir=out_path)
            response.success = result.get("status") == "success"
            response.message = json.dumps(
                {
                    "status": result.get("status"),
                    "failure_reason": result.get("failure_reason"),
                    "result_json": result.get("artifacts", {}).get("result_json"),
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            response.success = False
            response.message = json.dumps(
                {
                    "status": "failed_internal_error",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            )
        return response

    def spin(self) -> None:
        self._rclpy.spin(self.node)

    def destroy(self) -> None:
        self.node.destroy_node()


def main(argv: list[str] | None = None) -> int:
    args, ros_args = build_arg_parser().parse_known_args(argv)
    if args.check:
        return _run_check()

    import rclpy

    rclpy.init(args=ros_args)
    adapter = None
    try:
        adapter = LevelPlannerRosNode()
        adapter.spin()
        return 0
    finally:
        if adapter is not None:
            adapter.destroy()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

