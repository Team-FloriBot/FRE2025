#!/usr/bin/env python
import rospy
import math
from geometry_msgs.msg import Twist, Pose, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
import tf
import tf2_ros
from tf.transformations import quaternion_from_euler
import time

class FakeOdomNode:
    def __init__(self):
        rospy.init_node('fake_odom_from_cmdvel')

        # Publisher
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)

        # TF Broadcaster
        self.odom_broadcaster = tf2_ros.TransformBroadcaster()

        # Subscriber
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)

        # Initial Position
        self.x = 0.5
        self.y = 0.375
        self.th = 0.0  # theta (Yaw)

        # Velocity
        self.vx = 0.0
        self.vth = 0.0

        self.last_time = rospy.Time.now()

        rospy.Timer(rospy.Duration(0.1), self.update_odom)  # 50 Hz

    def cmd_vel_callback(self, msg):
        self.vx = msg.linear.x
        self.vth = msg.angular.z

    def update_odom(self, event):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Update position estimate
        delta_x = self.vx * math.cos(self.th) * dt
        delta_y = self.vx * math.sin(self.th) * dt
        delta_th = self.vth * dt

        self.x += delta_x
        self.y += delta_y
        self.th += delta_th

        # Create quaternion from yaw
        odom_quat = quaternion_from_euler(0, 0, self.th)

        # Publish TF
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = odom_quat[0]
        t.transform.rotation.y = odom_quat[1]
        t.transform.rotation.z = odom_quat[2]
        t.transform.rotation.w = odom_quat[3]
        self.odom_broadcaster.sendTransform(t)

        # Publish Odometry message
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = Quaternion(*odom_quat)

        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.angular.z = self.vth

        self.odom_pub.publish(odom)


if __name__ == '__main__':
    try:
        FakeOdomNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
