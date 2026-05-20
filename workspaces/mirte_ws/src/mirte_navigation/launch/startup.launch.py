from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    state_manager = Node(
        package='mirte_statemachine',
        executable='StateManager',
        name='StateManager',
        output='screen'
    )

    detector = Node(
        package='mirte_perception',
        executable='GoalGenerator',
        name='GoalGenerator',
        output='screen'
    )

    vision_controller = Node(
        package='mirte_navigation',
        executable='VisionController',
        name='VisionController',
        output='screen'
    )

    return LaunchDescription([
        state_manager,
        detector,
        vision_controller
    ])