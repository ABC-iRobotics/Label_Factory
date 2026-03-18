#!/usr/bin/env python3
import os 
import yaml
import subprocess
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

import rclpy.time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import numpy as np
from scipy.spatial.transform import Rotation as R

class FrameListener(Node):
    def __init__(self):
        super().__init__('ee_pose')
        result = subprocess.check_output("ros2 pkg prefix label_factory",shell = True, text = True)
        result = result.split("/install",1)[0]
        with open(os.path.join(result,'label_factory/config.yaml'), 'r') as file:
            self.config = yaml.safe_load(file)
        # Declare and acquire `target_frame` parameter
        self.target_frame = self.declare_parameter('target_frame', self.config['moveit_configs']['base_link_name']).get_parameter_value().string_value
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Create a PoseStamped publisher
        self.publisher = self.create_publisher(PoseStamped, 'ee_pose', 1)
        # Call on_timer function every second
        self.timer = self.create_timer(0.1, self.on_timer)

    def on_timer(self):
        # Store frame names in variables that will be used to
        # compute transformations
        from_frame_rel = self.target_frame
        to_frame_rel = self.config['moveit_configs']['end_effector_name']
        try:
            t = self.tf_buffer.lookup_transform(from_frame_rel, to_frame_rel, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return
        msg = PoseStamped()
        msg.pose.position.x = t.transform.translation.x
        msg.pose.position.y = t.transform.translation.y
        msg.pose.position.z = t.transform.translation.z
        msg.pose.orientation.x = t.transform.rotation.x
        msg.pose.orientation.y = t.transform.rotation.y
        msg.pose.orientation.z = t.transform.rotation.z
        msg.pose.orientation.w = t.transform.rotation.w
        clear = lambda: os.system('clear')
        clear()
        print("Pose:\nx: " + str(msg.pose.position.x) + "\ny: " + str(msg.pose.position.y) + "\nz: " + str(msg.pose.position.z)+ "\n---\nOrientation\nx:" + str(msg.pose.orientation.x) + "\ny:" + str(msg.pose.orientation.y) + "\nz: " + str(msg.pose.orientation.z) + "\nw:" + str(msg.pose.orientation.w) + "\n_________________________" ) 
        self.publisher.publish(msg)

def main():
    rclpy.init()
    node = FrameListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
