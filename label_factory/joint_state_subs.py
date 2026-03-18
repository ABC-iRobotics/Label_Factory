#!/usr/bin/env python3
from rclpy.node import Node
from sensor_msgs.msg import JointState

class Joint_state_Subscriber(Node):
    def __init__(self):
        super().__init__('Joints_subs')
        self.subscription = self.create_subscription(JointState,'/joint_states',self.listener_callback,10)
        self.subscription  # prevent unused variable warning
        self.msg = JointState()

    def listener_callback(self, msg):
        self.msg = msg