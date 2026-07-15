from setuptools import find_packages, setup


package_name = "level_planner_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[
        "level_planner",
        "level_planner.*",
        "level_planner_core",
        "level_planner_core.*",
        "level_planner_cli",
        "level_planner_cli.*",
        "level_planner_ros",
        "level_planner_ros.*",
    ]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/level_planner.launch.py"]),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="caohy",
    maintainer_email="caohy@example.com",
    description="Optional ROS adapter for the standalone SR5 level constrained planner",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "planner_node=level_planner_ros.planner_node:main",
        ],
    },
)

