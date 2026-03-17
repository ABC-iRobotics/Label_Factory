import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'label_factory'

setup(
    name = package_name,
    version = '0.0.0',
    packages = find_packages(exclude=['test']),
    data_files = [
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires = ['setuptools'],
    zip_safe = True,
    maintainer = 'Armin Karoly',
    maintainer_email = 'armin.karoly@irob.uni-obuda.hu',
    description = 'This package is an addition to the Blender Annotation Tool, to create real images with annotations.',
    license = 'MIT',
    # With this:
    extras_require={'test': ['pytest', 'mock'],},
    entry_points = {
        'console_scripts': [
            'ee_pose = label_factory.ee_pose:main',
            'UI = label_factory.UI:main',
            'ChArUco = label_factory.create_charuco_board:main',
            'Demo = label_factory.demo:main',
        ],
    },
)
