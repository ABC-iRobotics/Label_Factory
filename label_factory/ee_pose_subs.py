#!/usr/bin/env python3
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class EE_Pose_Subscriber(Node):
    def __init__(self):
        super().__init__('ee_pose_sub')
        self.subscription = self.create_subscription(PoseStamped,'ee_pose',self.listener_callback,10)
        self.subscription  # prevent unused variable warning
        self.ee_pose = PoseStamped()

    def listener_callback(self, msg):
        self.ee_pose = msg