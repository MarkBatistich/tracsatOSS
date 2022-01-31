import icp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# P = np.array([[1, 1, 3], [1, 3, 3]])
# Q = np.array([[1.5, 2.5, 4], [1.25, 2.5, 2]])

data = pd.read_csv('data/distance_0deg_0.csv', header=None)
data = data[0].astype(float)
rad0 = data.to_numpy()

data = pd.read_csv('data/ellen1.csv', header=None)
data = data[0].astype(float)
rad1 = data.to_numpy()

theta0 = (np.linspace(-45,225,811)*np.pi/180.0)
theta1 = theta0

plt.figure()
plt.polar(theta1,rad1,'b.')
plt.show()

theta0 = theta0[rad0 < 4.9]
rad0 = rad0[rad0 < 4.9]
print(theta0.size)
theta1 = theta1[rad1 < 4.9]
rad1 = rad1[rad1 < 4.9]

plt.figure()
plt.polar(theta0,rad0,'b.')


P = icp.createPointList(rad0, theta0)
Q = icp.createPointList(rad0, theta0)

th = 10 * np.pi / 180

C0 = np.array([[np.cos(th), -np.sin(th)],[np.sin(th),np.cos(th)]])

P = C0 @ P
C,T = icp.icp(P,Q)
angle = np.rad2deg(icp.angleOfRot(C))
print(angle)

P2 = C.T @ P + T
ax = icp.plotPoints(P, Q, "P", "Q")
ax = icp.plotPoints(P2, Q, label1='Pnew', label2='Q')

plt.show()
