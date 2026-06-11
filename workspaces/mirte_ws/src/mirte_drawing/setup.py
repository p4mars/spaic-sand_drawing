from setuptools import find_packages, setup

package_name = 'mirte_drawing'

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
    maintainer='wtoutenhoofd',
    maintainer_email='wtoutenhoofd.tudelft@gmail.com',
    description='Sand drawing module – controls arm/gripper to draw patterns in sand',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'SandDrawer = mirte_drawing.SandDrawer:main',
        ],
    },
)
