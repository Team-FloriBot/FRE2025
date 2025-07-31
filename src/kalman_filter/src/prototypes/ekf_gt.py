#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates

def callback(msg):
    try:
        idx = msg.name.index("floribot")
        gt_pose = msg.pose[idx]
        x = gt_pose.position.x
        y = gt_pose.position.y
        print(f"Ground Truth: x={x:.2f}, y={y:.2f}")
    except ValueError:
        print("Robotername nicht gefunden")
        
rospy.init_node("test_node")
rospy.Subscriber("/gazebo/model_states", ModelStates, callback)
rospy.spin()