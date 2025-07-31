#!/usr/bin/env python

'''
Executes a localization of the Floribot by integrating an Extended Kalman Filter. Uses the odometry 
for the prediction step and the lidar-scanner for the measurement step. Publishes the calculated pose
as topic "/ekf_pose" and saves it to a csv file.
'''

# ##### Imports #####
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int32, String
import math
import csv
import os

# EKF class inittialization
class EKF_Localization:
    def __init__(self):
        # Node initialization
        rospy.init_node("ekf_localization_node")

        # Initialization of csv file
        self.log_file_path = os.path.expanduser("/home/user/ekf_position_log.csv")
        self.log_file = open(self.log_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['time', 'x', 'y', 'theta', 'measurement_type'])
        self.measurement_type = "odom"

        self.row_width = 0.75
        self.theta_odom = 0
        self.row_dir = " "

        # Initial state [x, y, theta]
        self.x = np.array([[-0.5],[0.375],[0]])
        self.P = np.eye(3) * 0.1  # Anfangskovarianz
        
        self.row_index = 0
        self.timestamp_lidar = rospy.Time.now().to_sec()

        # Measurement matrix for x, y and theta
        self.H = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        # Process and measurement noise covariance
        self.Q = np.array([
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, np.deg2rad(15)]
        ])

        self.R_lidar = np.array([
            [0.005, 0, 0],
            [0, 0.005, 0],
            [0, 0, 0.01]
        ]) 

        # Subscriber
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/laser_scanner_front", LaserScan, self.scan_callback)
        rospy.Subscriber("/row_index", Int32, self.row_index_callback)
        rospy.Subscriber("/row_dir", String, self.row_dir_callback)
        rospy.Timer(rospy.Duration(0.1), self.prediction_step)

        # Publisher for the EKF-pose
        self.pose_pub = rospy.Publisher("/ekf_pose", PoseWithCovarianceStamped, queue_size=1)

        # Execute continously
        rospy.spin()

    def row_index_callback(self, msg):
        '''
        Callback function for row index (counter), coming from the navigation node.
        '''
        # Get data from incoming message
        self.row_index = msg.data

    def row_dir_callback(self, msg):
        '''
        Callback function for driving direction, coming from the navigation node.
        '''
        # Get data from incoming message
        self.row_dir = msg.data

    def prediction_step(self, event):
        '''
        Executes a prediction step using odometry data.
        '''
        # Set a time constant of 0.1s for calculation of the prediction step
        dt = 0.1

        # Calculation of the time step since the LIDAR-scan (interrupt), so that the state estimation is
        # more precise. However, this approach provides partially worse results than with a constant time step
        # (problems regarding latency?)

        # now = rospy.Time.now()
        # current_timestamp = now.to_sec()

        # if abs(current_timestamp - self.timestamp_lidar) < 0.1:
        #     dt = abs(current_timestamp - self.timestamp_lidar)
        # else:
        #     dt = 0.1

        if not hasattr(self, 'v') or not hasattr(self, 'omega'):
            return
        
        # Extract theta from the state vector to calculate x and y
        theta = self.x[2, 0]

        # Prediction step
        B = np.array([
            [np.cos(theta) * dt, 0],
            [np.sin(theta) * dt, 0],
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

        # Jump to publish function with measurement type "odom"
        self.measurement_type = "odom"
        self.publish_pose()

    def odom_callback(self, msg):
        '''
        Callback function for incoming odometry data.
        '''
        # Get data from incoming message
        self.last_odom_time = msg.header.stamp.to_sec()
        self.v = msg.twist.twist.linear.x
        self.omega = msg.twist.twist.angular.z

    def scan_callback(self, msg):       
        '''
        Callback function for incoming lidar-scan data.
        '''
        # Get data from incoming message
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # Left and right scans (±0.1 rad respectively)
        left_indices = range(int((np.pi/2 - 0.1 - angle_min) / angle_increment),
                            int((np.pi/2 + 0.1 - angle_min) / angle_increment))
        right_indices = range(int((-np.pi/2 - 0.1 - angle_min) / angle_increment),
                            int((-np.pi/2 + 0.1 - angle_min) / angle_increment))

        # Extract only valid measurements form the scans (with a specified length)
        valid_left = [ranges[i] for i in left_indices if 0 < ranges[i] < self.row_width]
        valid_right = [ranges[i] for i in right_indices if 0 < ranges[i] < self.row_width]

        # Check if there are valid measurements on the left and right
        if len(valid_left) < 1 or len(valid_right) < 1:
            return

        # Calculate the mean of those measurements
        left_dist = np.mean(valid_left)
        right_dist = np.mean(valid_right)

        # Calculate y and theta from valid scans to the left and right of the robot
        measured_y = self.row_width / 2.0 - (left_dist - right_dist) / 2.0
        measured_y += self.row_index
        measured_theta = np.arctan2(left_dist - right_dist, self.row_width)

        # Set the valid measurement length to 1.5 length of the row
        row_length = 10.0
        min_dist = row_length * 1.5
        min_dist_l = min_dist
        min_dist_r = min_dist

        # Search for the end of the row by looking for the first valid measurement
        # starting from the middle
        for i in range(int(len(ranges)/2), int(len(ranges))):
            if 0.0 < ranges[i] < min_dist:
                min_dist_l = ranges[i]
                break

        for i in range(int(len(ranges)/2), 0, -1):
            if 0.0 < ranges[i] < min_dist:
                min_dist_r = ranges[i]
                break

        min_dist = min(min_dist_l, min_dist_r)

        # Check which direction the robot is going to adjust theta and x
        if self.row_dir == "L":
            measured_theta_cor = measured_theta + np.pi
            measured_x = min_dist
        else:
            measured_theta_cor = measured_theta
            measured_x = row_length - min_dist

        # If the distance difference is too big (which implicates a wrong measurement), take the old value
        if abs(measured_x - self.x[0, 0]) > 2:
            measured_x = self.x[0, 0]

        # Update step
        z = np.array([[measured_x], [measured_y], [measured_theta_cor]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R_lidar
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P

        # Jump to publish function with measurement type "laser"
        self.measurement_type = "laser"
        self.publish_pose()

    def publish_pose(self):
        '''
        Publishes the calculated EKF-pose as a new topic and saves position data to a csv-file.
        '''
        # Extract pose components from the state vector
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose.pose.position.x = self.x[0, 0]
        msg.pose.pose.position.y = self.x[1, 0]
        msg.pose.pose.orientation.z = math.degrees(self.x[2, 0])

        # Get components
        cov = np.zeros((6, 6))
        cov[0, 0] = self.P[0, 0]
        cov[1, 1] = self.P[1, 1]
        cov[5, 5] = self.P[2, 2]
        msg.pose.covariance = cov.flatten().tolist()

        # Publish the pose
        self.pose_pub.publish(msg)

        # Saves timestamp and pose components in csv file
        timestamp = rospy.Time.now().to_sec()
        x = self.x[0, 0]
        y = self.x[1, 0]
        theta = self.x[2, 0]
        self.csv_writer.writerow([timestamp, x, y, theta, self.measurement_type])
        self.log_file.flush()


if __name__ == "__main__":
    try:
        EKF_Localization()
    except rospy.ROSInterruptException:
        pass
