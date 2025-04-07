import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Paths to packages
    tb3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')

    # Launch Gazebo simulation with the robot
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_dir, 'launch', 'empty_world.launch.py')
        )
    )

    # Custom teleop node
    teleop_node = Node(
        package='bachir_gemayel',
        executable='bachir_gemayel_node',
        name='bachir_gemayel',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'linear_scale': 1.0},
            {'angular_scale': 1.0},
            {'use_sim_time': True}
        ]
    )

    auto_stop = Node(
        package='mybot_control',
        executable='lds_distance',  # or 'auto_stop' if you named it differently
        name='lds_distance',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ]
    )

    return LaunchDescription([
        gazebo,
        teleop_node
        auto_stop
    ])
