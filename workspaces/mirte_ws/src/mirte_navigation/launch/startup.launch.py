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
        executable='ArucoDetector',
        name='ArucoDetector',
        output='screen'
    )

    white_board_tracker = Node(
        package='mirte_navigation',
        executable='WhiteBoardTracker',
        name='WhiteBoardTracker',
        output='screen'
    )

    sandpit_tracker = Node(
        package='mirte_perception',
        executable='ArucoDetectionSand',
        name='ArucoDetectionSand',
        output='screen'
    )

    board_reader = Node(
        package='mirte_perception',
        executable='TextDetector',
        name='TextDetector',
        output='screen'
    )

    sand_drawer = Node(
        package='mirte_drawing',
        executable='SandDrawer',
        name='SandDrawer',
        output='screen'
    )

    return LaunchDescription([
        state_manager,
        sandpit_tracker,
        white_board_tracker,
        detector,
        board_reader,
        sand_drawer
    ])