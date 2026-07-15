"""Compatibility entrypoint for the standalone planner CLI."""

from level_planner.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
