from setuptools import find_packages, setup

package_name = 'mirte_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Wout',
    maintainer_email='wbarrez@student.tudelft.nl',
    description='Determines the position of objects in the environment',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'GoalGenerator = mirte_perception.ObjectDetection:main',
            'ArucoDetector = mirte_perception.ArucoDetection:main',
        ],
    },
)
