import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'projet_ros_2025'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='turtle',
    maintainer_email='turtle@todo.todo',
    description='ROS2 project',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_following = projet_ros_2025.line_following:main',
            'teleop_camera = projet_ros_2025.teleop_camera:main',
            'object_avoidance = projet_ros_2025.object_avoidance:main',
            'corridor = projet_ros_2025.corridor:main',
            'foot = projet_ros_2025.Football:main',
            'vision = projet_ros_2025.vision:main'
        ],
    },
)
