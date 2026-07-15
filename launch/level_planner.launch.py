"""Launch only the optional standalone level planner ROS adapter."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value="configs/sr5_level.yaml"),
        DeclareLaunchArgument("request", default_value="examples/requests/request_level_001.json"),
        DeclareLaunchArgument("out_dir", default_value="runs/ros_adapter"),
        DeclareLaunchArgument("diffusion_seed_mode", default_value=""),
        DeclareLaunchArgument("load_planner_on_start", default_value="true"),
        Node(
            package="level_planner_ros",
            executable="planner_node",
            name="level_planner_ros",
            output="screen",
            parameters=[{
                "config": LaunchConfiguration("config"),
                "request": LaunchConfiguration("request"),
                "out_dir": LaunchConfiguration("out_dir"),
                "diffusion_seed_mode": LaunchConfiguration("diffusion_seed_mode"),
                "load_planner_on_start": LaunchConfiguration("load_planner_on_start"),
            }],
        ),
    ])

