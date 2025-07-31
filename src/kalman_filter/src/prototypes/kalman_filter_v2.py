#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int32
import math

class EKF_Localization:
    def __init__(self):
        rospy.init_node("ekf_localization_node")

        # Initial state [x, y, theta]
        self.x = np.zeros((3, 1))
        self.P = np.eye(3) * 0.1  # Anfangskovarianz
        
        self.row_index = 0

        # Prozess- und Messrauschen
        self.Q = np.diag([0.05, 0.05, np.deg2rad(1)])  # Prozessrauschen
        self.R_lidar = np.diag([0.1, 0.1])  # Messrauschen der LIDAR-Abstände (links, rechts)

        # Abonnenten
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/laser_scanner_front", LaserScan, self.scan_callback)
        rospy.Subscriber("/row_index", Int32, self.row_index_callback)

        # Publisher für geschätzte Pose
        self.pose_pub = rospy.Publisher("/ekf_pose", PoseWithCovarianceStamped, queue_size=1)

        rospy.spin()

    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def row_index_callback(self, msg):
        self.row_index = msg.data

    def odom_callback(self, msg):
        # Nehme Geschwindigkeit als Eingabe u = [v, omega]
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        dt = 0.1  # Simpler Zeitinkrement, alternativ msg.header.stamp verwenden

        theta = self.x[2, 0]

        # EKF Prediction
        F = np.eye(3)
        B = np.array([
            [np.cos(theta)*dt, 0],
            [np.sin(theta)*dt, 0],
            [0, dt]
        ])
        u = np.array([[v], [omega]])
        self.x = self.x + B @ u

        # Jacobian of motion model w.r.t state
        Fx = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0, 1]
        ])
        self.P = Fx @ self.P @ Fx.T + self.Q

    def scan_callback(self, msg):
        # Extrahiere linken und rechten Abstand zu Pflanzenreihen
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        # Linker und rechter Strahlbereich
        left_indices = range(int((np.pi/2 - 0.1 - angle_min) / angle_increment), int((np.pi/2 + 0.1 - angle_min) / angle_increment))
        right_indices = range(int((-np.pi/2 - 0.1 - angle_min) / angle_increment), int((-np.pi/2 + 0.1 - angle_min) / angle_increment))

        # Mittelwerte
        left_dist = np.mean([ranges[i] for i in left_indices if 0 < ranges[i] < msg.range_max])
        right_dist = np.mean([ranges[i] for i in right_indices if 0 < ranges[i] < msg.range_max])

        if np.isnan(left_dist) or np.isnan(right_dist):
            return  # Ungültige Daten ignorieren

        # EKF Update: Messen, wo der Roboter innerhalb der Pflanzenreihe ist (z. B. y-Position und Orientierung)
        row_width = 1.0  # bekannte Breite der Pflanzenreihe (z. B. 1 m)
        measured_y = row_width / 2.0 - (left_dist - right_dist) / 2.0
        measured_y += self.row_index 
        measured_theta = np.arctan2(left_dist - right_dist, row_width)

        z = np.array([[measured_y], [measured_theta]])

        # Messmatrix H: misst y und theta
        H = np.array([
            [0, 1, 0],
            [0, 0, 1]
        ])

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_lidar
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

        self.publish_pose()

    def publish_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = self.x[0, 0]
        msg.pose.pose.position.y = self.x[1, 0]
        msg.pose.pose.orientation.z = math.degrees(np.sin(self.x[2, 0]/2.0))
        # msg.pose.pose.orientation.w = np.cos(self.x[2, 0]/2.0)

        # Flatten covariance
        cov = np.zeros((6, 6))
        cov[0, 0] = self.P[0, 0]
        cov[1, 1] = self.P[1, 1]
        cov[5, 5] = self.P[2, 2]
        msg.pose.covariance = cov.flatten().tolist()

        self.pose_pub.publish(msg)

if __name__ == "__main__":
    try:
        EKF_Localization()
    except rospy.ROSInterruptException:
        pass
