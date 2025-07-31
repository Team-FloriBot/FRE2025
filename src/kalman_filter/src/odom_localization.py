#!/usr/bin/env python

import rospy
import numpy as np
import math
import csv
import os
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

class OdomLocalization:
    def __init__(self):
        rospy.init_node("odom_localization_node")

        # Log-Datei vorbereiten
        self.log_file_path = os.path.expanduser("/home/user/odom_localization_log.csv")
        self.log_file = open(self.log_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['time', 'x', 'y', 'theta'])

        # Initialzustand [x, y, theta]
        self.x = np.zeros((3, 1))
        self.P = np.eye(3) * 0.1

        # Rauschen des Bewegungsmodells
        self.Q = np.array([
            [0.0001, 0, 0],
            [0, 0.0001, 0],
            [0, 0, 0.0001]
        ])

        self.last_time = None
        self.v = 0.0
        self.omega = 0.0

        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        rospy.Timer(rospy.Duration(0.5), self.prediction_step)
        rospy.spin()

    def odom_callback(self, msg):
        self.v = msg.twist.twist.linear.x
        self.omega = msg.twist.twist.angular.z
        self.last_time = msg.header.stamp.to_sec()

    def prediction_step(self, event):
        if self.last_time is None:
            return

        dt = 0.5  # 10 Hz

        theta = self.x[2, 0]

        # Bewegungsvorhersage
        B = np.array([
            [np.cos(theta)*dt, 0],
            [np.sin(theta)*dt, 0],
            [0, dt]
        ])
        u = np.array([[self.v], [self.omega]])
        self.x = self.x + B @ u

        Fx = np.array([
            [1, 0, -self.v * np.sin(theta) * dt],
            [0, 1,  self.v * np.cos(theta) * dt],
            [0, 0, 1]
        ])
        self.P = Fx @ self.P @ Fx.T + self.Q

        self.publish_pose()

    def publish_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "odom"

        msg.pose.pose.position.x = self.x[0, 0]
        msg.pose.pose.position.y = self.x[1, 0]
        msg.pose.pose.orientation.z = math.sin(self.x[2, 0] / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.x[2, 0] / 2.0)

        # Nur relevante Kovarianzen setzen
        cov = np.zeros((6, 6))
        cov[0, 0] = self.P[0, 0]
        cov[1, 1] = self.P[1, 1]
        cov[5, 5] = self.P[2, 2]
        msg.pose.covariance = cov.flatten().tolist()

        timestamp = rospy.Time.now().to_sec()
        x, y, theta = self.x.flatten()
        self.csv_writer.writerow([timestamp, x, y, theta])
        self.log_file.flush()

if __name__ == "__main__":
    try:
        OdomLocalization()
    except rospy.ROSInterruptException:
        pass