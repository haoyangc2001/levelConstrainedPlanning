"""Checkpoint-backed diffusion seed provider and critic selector."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .condition import build_condition_tensor
from .seed_provider import SeedCandidate, SeedProviderResult


@dataclass
class CheckpointDiffusionSeedProviderConfig:
    mode: str = "off"
    diffusion_checkpoint_path: str = ""
    critic_checkpoint_path: str = ""
    k_generate: int = 8
    k_accept: int = 2
    timeout_sec: float = 2.0
    device: str = "cuda:0"
    max_step_l2: float = 1.0
    max_start_gap_l2: float = 0.05
    joint_abs_limit: float = 2.0 * math.pi
    use_critic: bool = True


class CheckpointDiffusionSeedProvider:
    provider_name = "diffusion_seed"

    def __init__(self, config: CheckpointDiffusionSeedProviderConfig) -> None:
        self.config = config

    def generate(self, request_context: dict[str, Any]) -> SeedProviderResult:
        started = time.time()
        mode = str(self.config.mode or "off").strip().lower()
        metadata: dict[str, Any] = {
            "phase": "closed_loop.phase5_checkpoint_diffusion",
            "source_kind": "checkpoint_diffusion_runtime",
            "mode": mode,
            "diffusion_checkpoint_path": self.config.diffusion_checkpoint_path,
            "critic_checkpoint_path": self.config.critic_checkpoint_path,
            "k_generate": int(self.config.k_generate),
            "k_accept": int(self.config.k_accept),
            "timeout_sec": float(self.config.timeout_sec),
            "use_critic": bool(self.config.use_critic),
        }
        if mode == "off":
            return SeedProviderResult(self.provider_name, mode, "disabled", metadata=metadata)
        checkpoint_path = Path(self.config.diffusion_checkpoint_path)
        if not checkpoint_path.exists():
            return SeedProviderResult(
                self.provider_name,
                mode,
                "checkpoint_missing",
                metadata=metadata,
                error=f"diffusion_checkpoint_not_found:{checkpoint_path}",
            )
        try:
            candidates, generation_metadata = self._sample_candidates(request_context, started)
            metadata.update(generation_metadata)
        except Exception as exc:
            return SeedProviderResult(
                self.provider_name,
                mode,
                "generation_failed",
                candidates=[],
                metadata={**metadata, "elapsed_sec": round(time.time() - started, 6)},
                error=f"{type(exc).__name__}: {exc}",
            )
        if self.config.use_critic and candidates:
            try:
                candidates, critic_metadata = self._select_with_critic(candidates, request_context)
                metadata.update(critic_metadata)
            except Exception as exc:
                metadata["critic_status"] = "critic_failed_fallback_to_precheck_order"
                metadata["critic_error"] = f"{type(exc).__name__}: {exc}"
                candidates = candidates[: max(1, int(self.config.k_accept))]
        else:
            metadata["critic_status"] = "disabled_or_no_candidates"
            candidates = candidates[: max(1, int(self.config.k_accept))]
        metadata["elapsed_sec"] = round(time.time() - started, 6)
        metadata["accepted_count"] = len(candidates)
        return SeedProviderResult(
            provider_name=self.provider_name,
            mode=mode,
            status="generated" if candidates else "no_generated_candidates",
            candidates=candidates,
            metadata=metadata,
        )

    def _sample_candidates(
        self,
        request_context: dict[str, Any],
        started: float,
    ) -> tuple[list[SeedCandidate], dict[str, Any]]:
        from tools.learning.diffusion_seed_learning.dataset import Normalization
        from tools.learning.diffusion_seed_learning.diffusion import GaussianDiffusion1D
        from tools.learning.diffusion_seed_learning.model_unet1d import TemporalUNet1D

        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.config.diffusion_checkpoint_path, map_location=device)
        model = TemporalUNet1D(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        diffusion = GaussianDiffusion1D(**checkpoint["diffusion_config"]).to(device)
        normalization = Normalization.from_json(checkpoint["normalization"])
        k_generate = max(1, int(self.config.k_generate))
        condition = build_condition_tensor(request_context).to(device).unsqueeze(0).repeat(k_generate, 1)
        with torch.no_grad():
            normalized = diffusion.sample(
                model,
                (k_generate, int(checkpoint["horizon"]), int(checkpoint["model_config"]["dof"])),
                condition,
            ).cpu()
        trajectories = normalization.denormalize(normalized)
        q_start = torch.tensor(request_context.get("start_joint") or [0.0] * trajectories.shape[-1], dtype=torch.float32)
        trajectories = _recover_continuity(
            trajectories,
            q_start,
            max_step_l2=float(self.config.max_step_l2),
            joint_abs_limit=float(self.config.joint_abs_limit),
        )
        candidates: list[SeedCandidate] = []
        for index in range(int(trajectories.shape[0])):
            if time.time() - started > float(self.config.timeout_sec):
                break
            points = [[round(float(v), 8) for v in row] for row in trajectories[index].tolist()]
            precheck = _precheck_generated_seed(
                points,
                start_joint=q_start.tolist(),
                max_start_gap_l2=float(self.config.max_start_gap_l2),
                max_step_l2=float(self.config.max_step_l2),
                joint_abs_limit=float(self.config.joint_abs_limit),
            )
            candidates.append(
                SeedCandidate(
                    candidate_id=f"diffusion_seed_{index:02d}",
                    source_label=f"diffusion_seed_{index:02d}",
                    source_type="diffusion",
                    optimized=False,
                    entered_pool=False,
                    trajectory_points=points,
                    metadata={
                        "provider": self.provider_name,
                        "model_version": str(self.config.diffusion_checkpoint_path),
                        "checkpoint_schema_version": checkpoint.get("schema_version"),
                        "horizon": int(checkpoint["horizon"]),
                        "condition_dim": int(condition.shape[-1]),
                    },
                    precheck=precheck,
                    metrics={
                        "raw_seed_max_step_jump_l2": precheck.get("joint_step_max_l2"),
                    },
                )
            )
        return candidates, {
            "checkpoint_schema_version": checkpoint.get("schema_version"),
            "horizon": int(checkpoint["horizon"]),
            "available_generated_count": int(len(candidates)),
            "precheck_valid_count": int(sum(1 for c in candidates if c.precheck.get("valid"))),
        }

    def _select_with_critic(
        self,
        candidates: list[SeedCandidate],
        request_context: dict[str, Any],
    ) -> tuple[list[SeedCandidate], dict[str, Any]]:
        from tools.learning.diffusion_seed_learning.critic import (
            CriticNormalization,
            SuccessCriticMLP,
            build_critic_feature,
        )

        critic_path = Path(self.config.critic_checkpoint_path)
        if not critic_path.exists():
            return candidates[: max(1, int(self.config.k_accept))], {
                "critic_status": "critic_checkpoint_missing_fallback_to_precheck_order",
            }
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(critic_path, map_location=device)
        model = SuccessCriticMLP(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        normalization = CriticNormalization.from_json(checkpoint["normalization"])
        horizon = int(checkpoint.get("horizon", 32))
        rows: list[dict[str, Any]] = []
        features = []
        for index, candidate in enumerate(candidates):
            sample = _candidate_to_sample(candidate, request_context, index)
            rows.append(sample)
            features.append(build_critic_feature(sample, horizon))
        feature = normalization.normalize_feature(torch.stack(features, dim=0)).to(device)
        with torch.no_grad():
            output = model(feature)
            p_success = torch.sigmoid(output["success_logit"]).detach().cpu()
            regression = normalization.denormalize_regression(output["regression"].detach().cpu())
        scored: list[tuple[float, int, dict[str, float]]] = []
        for index in range(len(candidates)):
            alignment_risk, collision_risk, joint_jump_risk, expected_opt_time_ms = regression[index].tolist()
            quality = (
                float(p_success[index].item())
                - 0.20 * float(alignment_risk)
                - 0.10 * float(collision_risk)
                - 0.15 * float(joint_jump_risk)
                - 0.0005 * float(max(expected_opt_time_ms, 0.0))
            )
            scored.append(
                (
                    quality,
                    index,
                    {
                        "p_success": float(p_success[index].item()),
                        "alignment_risk": float(alignment_risk),
                        "collision_risk": float(collision_risk),
                        "joint_jump_risk": float(joint_jump_risk),
                        "expected_opt_time_ms": float(expected_opt_time_ms),
                        "quality_score": float(quality),
                    },
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        for rank, (_, index, score) in enumerate(scored[: max(1, int(self.config.k_accept))]):
            candidate = candidates[index]
            candidate.metadata["critic_score"] = score
            candidate.metadata["critic_rank"] = int(rank)
            selected.append(candidate)
        return selected, {
            "critic_status": "scored",
            "critic_checkpoint_path": str(critic_path),
            "critic_scored_count": len(scored),
            "critic_selected_count": len(selected),
        }


def _candidate_to_sample(candidate: SeedCandidate, request_context: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "sample_id": f"runtime_candidate_{index}",
        "sample_type": "candidate",
        "task": {
            "target_pose": request_context.get("target_pose"),
            "alignment": request_context.get("alignment") or {},
        },
        "start_state": {
            "service_start_joint": request_context.get("start_joint") or [],
        },
        "obstacle_world": request_context.get("world_summary") or {},
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "source_type": candidate.source_type,
            "trajectory": {
                "points": candidate.trajectory_points,
            },
            "metrics": candidate.metrics,
        },
        "source_lineage": {
            "source_type": candidate.source_type,
        },
        "labels": {
            "validator_valid": False,
            "positive_for_critic": False,
        },
    }


def _recover_continuity(
    trajectories: torch.Tensor,
    q_start: torch.Tensor,
    *,
    max_step_l2: float,
    joint_abs_limit: float,
) -> torch.Tensor:
    recovered = trajectories.clone()
    recovered[:, 0, :] = q_start
    for batch_index in range(recovered.shape[0]):
        for point_index in range(1, recovered.shape[1]):
            prev = recovered[batch_index, point_index - 1]
            current = recovered[batch_index, point_index]
            delta = current - prev
            norm = torch.linalg.norm(delta)
            if float(norm.item()) > float(max_step_l2):
                delta = delta / norm.clamp_min(1e-8) * float(max_step_l2)
            recovered[batch_index, point_index] = prev + delta
    return torch.clamp(recovered, -float(joint_abs_limit), float(joint_abs_limit))


def _precheck_generated_seed(
    trajectory: list[list[float]],
    *,
    start_joint: list[float],
    max_start_gap_l2: float,
    max_step_l2: float,
    joint_abs_limit: float,
) -> dict[str, Any]:
    if not trajectory:
        return {"valid": False, "failure_reason": "invalid_shape_or_dof"}
    points = torch.tensor(trajectory, dtype=torch.float32)
    start = torch.tensor(start_joint, dtype=torch.float32)
    start_gap = float(torch.linalg.norm(points[0] - start).item())
    step_l2 = torch.linalg.norm(points[1:] - points[:-1], dim=-1) if int(points.shape[0]) > 1 else torch.zeros(1)
    max_step = float(torch.max(step_l2).item()) if step_l2.numel() else 0.0
    max_abs = float(torch.max(torch.abs(points)).item())
    valid = (
        start_gap <= float(max_start_gap_l2)
        and max_step <= float(max_step_l2)
        and max_abs <= float(joint_abs_limit)
    )
    failure_reason = None
    if start_gap > float(max_start_gap_l2):
        failure_reason = "start_gap_exceeds_threshold"
    elif max_step > float(max_step_l2):
        failure_reason = "seed_step_exceeds_threshold"
    elif max_abs > float(joint_abs_limit):
        failure_reason = "joint_abs_limit_exceeded"
    return {
        "valid": bool(valid),
        "shape_valid": True,
        "finite": bool(torch.isfinite(points).all().item()),
        "start_gap_l2": round(start_gap, 8),
        "joint_step_max_l2": round(max_step, 8),
        "joint_abs_max": round(max_abs, 8),
        "thresholds": {
            "max_start_gap_l2": float(max_start_gap_l2),
            "max_step_l2": float(max_step_l2),
            "joint_abs_limit": float(joint_abs_limit),
        },
        "failure_reason": failure_reason,
    }
