#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math
from filterpy.kalman import ExtendedKalmanFilter
from functools import partial

class EKFLocalizer:
    def __init__(self):
        rospy.init_node("ekf_filterpy_localizer")

        # EKF mit Zustand x = [x, y, theta]
        self.ekf = ExtendedKalmanFilter(dim_x=3, dim_z=2)
        self.ekf.x = np.array([0., 0., 0.])  # initialer Zustand
        self.ekf.P *= 0.01                   # Unsicherheit im Anfangszustand    default 0.5
        self.ekf.R = np.diag([0.01, 0.01])    # Messrauschen   default 0.5
        self.ekf.Q = np.diag([0.05, 0.05, 0.01])  # Prozessrauschen

        self.last_time = None
        self.v = 0.0
        self.w = 0.0

        # ROS-Subscriber
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        # rospy.Subscriber("/laser_scanner_front", LaserScan, self.scan_callback)

    def odom_callback(self, msg):
        # Zeitdifferenz berechnen
        current_time = msg.header.stamp.to_sec()
        if self.last_time is None:
            self.last_time = current_time
            return
        dt = current_time - self.last_time
        self.last_time = current_time

        # Geschwindigkeiten auslesen
        self.v = msg.twist.twist.linear.x      # Lineargeschwindigkeit aus Odometrie
        self.w = msg.twist.twist.angular.z     # Winkelgeschwindigkeit aus Odometrie

        # rospy.loginfo("x: %.2f | θ: %.2f°", self.v, self.w,)

        # EKF-Vorhersage
        self.predict_step(dt)  # Position predicten

    def scan_callback(self, msg):
        theta = self.ekf.x[2]
        x_robot = self.ekf.x[0]
        y_robot = self.ekf.x[1]

        # Zwei Listen für zy-Werte von linker und rechter Seite
        left_y_measurements = []
        right_y_measurements = []

        for i, r in enumerate(msg.ranges):
            if np.isinf(r) or np.isnan(r):
                continue

            angle = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle)

            # Punkte nahe ±90° (seitlich)
            if 80 <= angle_deg <= 100:
                # Linke Seite
                local_x = r * math.cos(angle)
                local_y = r * math.sin(angle)
                zx = x_robot + local_x * math.cos(theta) - local_y * math.sin(theta)
                zy = y_robot + local_x * math.sin(theta) + local_y * math.cos(theta)
                left_y_measurements.append(zy)

            elif -100 <= angle_deg <= -80:
                # Rechte Seite
                local_x = r * math.cos(angle)
                local_y = r * math.sin(angle)
                zx = x_robot + local_x * math.cos(theta) - local_y * math.sin(theta)
                zy = y_robot + local_x * math.sin(theta) + local_y * math.cos(theta)
                right_y_measurements.append(zy)

        # Prüfe Mindestanzahl
        if len(left_y_measurements) >= 3 and len(right_y_measurements) >= 3:
            # Kombinierte Messungen
            all_y = left_y_measurements + right_y_measurements
            y_mean = np.mean(all_y)
            z = np.array([y_mean])
            self.ekf.update(z, self.H_jacobian, self.hx)

    # Bewegungsmodell (anpassbar!)
    def fx(self, x, dt):
        theta = x[2]
        v, w = self.v, self.w
        if abs(w) < 1e-5:
            dx = v * math.cos(theta) * dt
            dy = v * math.sin(theta) * dt
            dtheta = 0
        else:
            dx = (v / w) * (math.sin(theta + w * dt) - math.sin(theta))
            dy = (v / w) * (-math.cos(theta + w * dt) + math.cos(theta))
            # dtheta = w * dt
            dtheta = 0

        x_new = x + np.array([dx, dy, dtheta])
        x_new[2] = self.normalize_angle(x_new[2])

        rospy.loginfo("x: %.2f | y: %.2f | θ: %.2f°", x_new[0], x_new[1], math.degrees(x_new[2]))

        return x_new

    # Jacobian des Bewegungsmodells
    def F_jacobian(self, x, dt):
        theta = x[2]
        v, w = self.v, self.w
        if abs(w) < 1e-5:
            return np.array([
                [1, 0, -v * math.sin(theta) * dt],
                [0, 1,  v * math.cos(theta) * dt],
                [0, 0, 1]
            ])
        else:
            return np.array([
                [1, 0, (v / w) * (math.cos(theta + w * dt) - math.cos(theta))],
                [0, 1, (v / w) * (math.sin(theta + w * dt) - math.sin(theta))],
                [0, 0, 1]
            ])

    # Messfunktion: wir messen Position (x, y)
    def hx(self, x):
        return np.array([x[0], x[1]])

    # Jacobian der Messfunktion
    def H_jacobian(self, x):
        return np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])

    def predict_step(self, dt):
        # Manuelle Vorhersage des neuen Zustands
        self.ekf.x = self.fx(self.ekf.x, dt)

        # Neue Zustandskovarianz berechnen mit F
        self.ekf.F = self.F_jacobian(self.ekf.x, dt)
        self.ekf.P = np.dot(self.ekf.F, np.dot(self.ekf.P, self.ekf.F.T)) + self.ekf.Q


    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

if __name__ == "__main__":
    try:
        EKFLocalizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
