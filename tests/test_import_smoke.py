def test_core_import_smoke():
    import level_planner_core

    assert hasattr(level_planner_core, "LevelConstrainedPlanner")

