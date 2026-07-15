"""Core API for SR5 level constrained planning."""

__all__ = [
    "LevelConstrainedPlanner",
]


class LevelConstrainedPlanner:
    """Placeholder until phase 3 extracts the CuRobo-backed planner core."""

    @classmethod
    def from_config(cls, config_path):
        raise NotImplementedError("LevelConstrainedPlanner is implemented in phase 3.")

    def plan(self, request):
        raise NotImplementedError("LevelConstrainedPlanner is implemented in phase 3.")

