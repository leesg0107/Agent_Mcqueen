from setuptools import setup
import os
from glob import glob

package_name = 'agent_mcqueen'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leesg17',
    maintainer_email='leesg17@example.com',
    description='RL-based racing agent for F1Tenth simulation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stage1_agent = agent_mcqueen.stage1_agent_node:main',
            'overtake_agent = agent_mcqueen.overtake_agent_node:main',
        ],
    },
)
