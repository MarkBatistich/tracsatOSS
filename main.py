import icp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# P = np.array([[1, 1, 3], [1, 3, 3]])
# Q = np.array([[1.5, 2.5, 4], [1.25, 2.5, 2]])

data = pd.read_csv('data/distance_0deg_0.csv', header=None)
data = data[0].astype(float)
rad0 = data.to_numpy()

data = pd.read_csv('data/distance_0deg_1.csv', header=None)
data = data[0].astype(float)
rad1 = data.to_numpy()

theta = (np.linspace(-45,225,811)*np.pi/180.0).tolist()

plt.figure()
plt.polar(theta,rad0,'b.')
plt.figure()
plt.polar(theta,rad1,'b.')

P = icp.createPointList(rad1, theta)
Q = icp.createPointList(rad0, theta)
C,T = icp.icp(P,Q)
print(C)
angle = np.rad2deg(icp.angleOfRot(C))
print(angle);
print(T)



P2 = C.T @ P + T
ax = icp.plotPoints(P, Q, "P", "Q")
ax = icp.plotPoints(P2, Q, label1='Pnew', label2='Q')

plt.show()
