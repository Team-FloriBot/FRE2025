#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd

plt.figure()

# EKF localization
df = pd.read_csv('/home/user/ekf_position_log.csv')
df_laser = df[df['measurement_type'] == 'laser']
df_odom = df[df['measurement_type'] == 'odom']
plt.plot(df_odom['x'], df_odom['y'], 'b.', label='Odom prediction')
plt.plot(df_laser['x'], df_laser['y'], 'r.', label='Laser correction')
# plt.plot(df_odom['time'], df_odom['x'], 'b.', label='Odom prediction')
# plt.plot(df_laser['time'], df_laser['x'], 'r.', label='Laser correction')
plt.title('EKF Trajectory')

# Odometry localization only
# df = pd.read_csv('/home/user/odom_localization_log.csv')
# plt.plot(df['x'], df['y'], 'b.', label='Odom localization')
# # plt.plot(df['time'], df['x'], 'b.', label='Odom localization')
# plt.title('Position Trajectory (Odometry)')

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()