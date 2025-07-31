#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

class CmdVelFilter:
    def __init__(self):
        rospy.init_node('cmd_vel_filter_node')

        # self.alpha = rospy.get_param("~alpha", 0.5)
        self.alpha = rospy.get_param("alpha_reg")
        self.max_angular_change = 0.2

        self.prev_cmd = Twist()

        rospy.Subscriber("/cmd_vel_raw", Twist, self.cmd_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    def cmd_callback(self, msg):
        filtered = Twist()

        filtered.linear.x = self.alpha * msg.linear.x + (1 - self.alpha) * self.prev_cmd.linear.x
        filtered.angular.z = self.alpha * msg.angular.z + (1 - self.alpha) * self.prev_cmd.angular.z

        self.prev_cmd = filtered
        self.pub.publish(filtered)

if __name__ == "__main__":
    CmdVelFilter()
    rospy.spin()
