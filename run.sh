#!/bin/bash 

source /opt/ros/jazzy/setup.bash
source install/setup.bash

source .LF_venv/bin/activate
export PYTHONPATH=".LF_venv/lib/python3.12/site-packages:$PYTHONPATH"

ros2 run label_factory UI
