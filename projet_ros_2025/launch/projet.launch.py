from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the projet2025 world launch file
    projet2025_launch_path = os.path.join(
        get_package_share_directory('projet2025'),
        'launch',
        'projet.launch.py'
    )

    return LaunchDescription([
        # Include the projet2025 simulation world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(projet2025_launch_path)
        ),

        # Node for teleoperation
        Node(
            package='mybot_teleop',
            executable='turtlebot_teleop_node',
            parameters = [{
                'linear_scale':2.,
                'angular_scale':2.,
            }],
            name='turtlebot_teleop_node',
            output='screen'
        ),

        # Node for line following
        # Node(
        #     package='projet_ros_2025',
        #     executable='line_following',
        #     name='line_following_node',
        #     output='screen',
        #     parameters=[{'roundabout_direction': 'left'}]  # tu peux changer par 'right'
        # ),
    ])
