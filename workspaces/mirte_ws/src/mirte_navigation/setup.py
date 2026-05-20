from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mirte_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', 'mirte_navigation', 'launch'),
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wout',
    maintainer_email='wbarrez@student.tudelft.nl',
    description='Navigation stack for the Spatial AI project,',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'GoToGoal = mirte_navigation.go_to_relative_goal:main',
            'CrabWalk = mirte_navigation.crabwalk:main',
            'VisionDrive = mirte_navigation.vision_goal_controller:main',
            'VisionController = mirte_navigation.VisionController:main',

        ],
    },
)
