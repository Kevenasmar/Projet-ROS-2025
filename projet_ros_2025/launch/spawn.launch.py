#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Joep Tool

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, ExecuteProcess
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution, PathJoinSubstitution



def generate_launch_description():

    # On definit les listes des positions pour chaque challenge pour simplifier le lancement de la simu
    # Pour utiliser, on remplace simplement la liste pour le numero du challenge
    challenge_1=[0.84,-0.05,1.5708]
    challenge_2=[-0.8,0.75,-1.7708]
    challenge_3=[-0.825,-0.5,-1.5708]
    challenge_4=[-0.05,-0.22,0.0]
   

    model_path = os.path.join(os.path.expanduser("~"), "ros2_ws/src/projet2025/models/")
    spawn_random_ball_cmd =ExecuteProcess(
        cmd=["python3", os.path.join(model_path, "Ball/spawn_random_ball.py")],
        output="screen"
    )
   

    spawn_random_goal_cmd =ExecuteProcess(
        cmd=["python3", os.path.join(model_path, "robocup_3Dsim_goal/spawn_random_goal.py")],
        output="screen"
    )
   

    launch_file_dir = os.path.join(get_package_share_directory('projet2025'), 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_projet2025 = get_package_share_directory('projet2025')

    x_pose_arg = DeclareLaunchArgument(
        'x_pose', default_value=str(challenge_2[0]), #a changer selon le challenge
        description='x coordinate of spawned robot'
    )

    y_pose_arg = DeclareLaunchArgument(
        'y_pose', default_value=str(challenge_2[1]), #a changer selon le challenge
        description='y coordinate of spawned robot'
    )

    yaw_angle_arg = DeclareLaunchArgument(
        'yaw_angle', default_value=str(challenge_2[2]), #a changer selon le challenge
        description='yaw angle of spawned robot'
    )

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file = LaunchConfiguration('world', default='projet.sdf')

    set_env_vars_resources = AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(get_package_share_directory('projet2025'),
                         'models'))

    gazebo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': [PathJoinSubstitution([
            pkg_projet2025,
            'worlds',
            world_file
        ]),
        #TextSubstitution(text=' -r -v -v1 --render-engine ogre --render-engine-gui-api-backend opengl')],
        TextSubstitution(text=' -r -v -v1')],
        'on_exit_shutdown': 'true'}.items()
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': LaunchConfiguration('x_pose'),
            'y_pose': LaunchConfiguration('y_pose'),
            'yaw_angle': LaunchConfiguration('yaw_angle'),
        }.items()
    )

    ld = LaunchDescription()

    # Add the commands to the launch description
   
    ld.add_action(spawn_random_ball_cmd)
    ld.add_action(spawn_random_goal_cmd)
    ld.add_action(x_pose_arg)
    ld.add_action(y_pose_arg)
    ld.add_action(yaw_angle_arg)
    ld.add_action(set_env_vars_resources)
    ld.add_action(gazebo_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)

    return ld
