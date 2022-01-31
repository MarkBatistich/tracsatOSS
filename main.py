import icp
import filters as f
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# P = np.array([[1, 1, 3], [1, 3, 3]])
# Q = np.array([[1.5, 2.5, 4], [1.25, 2.5, 2]])

data = pd.read_csv('icpData/icp2.csv', header=None)
data = data[0].astype(float)
rad1 = data.to_numpy()

data = pd.read_csv('icpData/icp3.csv', header=None)
data = data[0].astype(float)
rad2 = data.to_numpy()

data = pd.read_csv('icpData/icp3.csv', header=None)
data = data[0].astype(float)
rad3 = data.to_numpy()

theta1 = (np.linspace(-45,225,811)*np.pi/180.0)
theta2 = theta1
theta3 = theta2

# plt.figure()
# plt.polar(theta1,rad1,'b.')
#plt.show()

theta1 = theta1[rad1 < 2]
rad1 = rad1[rad1 < 2]
rad1,theta1 = f.removeExtremeRad(rad1, theta1, 0.1, 1.5)
rad1,theta1 = f.removeExtremeAng(rad1, theta1, np.radians(45), np.radians(180))
print(theta1.size)
rad2,theta2 = f.removeExtremeRad(rad2, theta2, 0.1, 1.5)
rad2,theta2 = f.removeExtremeAng(rad2, theta2, np.radians(0), np.radians(180))

rad3,theta3 = f.removeExtremeRad(rad3, theta3, 0.1, 1.7)

# plt.figure()
# plt.polar(theta1,rad1,'b.')


P = icp.createPointList(rad2, theta2)
Q = icp.createPointList(rad1, theta1)

Pk = KMeans(n_clusters=2).fit(P.T).cluster_centers_.T
Qk = KMeans(n_clusters=2).fit(Q.T).cluster_centers_.T
# print(Pk)
# plt.figure()
# plt.plot(P[0,:],P[1,:],'b.')
# plt.plot(Pk[0,:],Pk[1,:],'bx')
# plt.axis('equal')
# plt.show()


#th = 10 * np.pi / 180
#C0 = np.array([[np.cos(th), -np.sin(th)],[np.sin(th),np.cos(th)]])
#P = C0 @ P

C,T = icp.icp(Pk,Qk)
angle = np.rad2deg(icp.angleOfRot(C))
print(angle)
print(T)
P2 = C.T @ P + T
Pk2 = C.T @ Pk + T
ax = icp.plotPointsK(P, Q, Pk, Qk, "t2", "t1")
plt.title('Lidar Data and Clusters')
ax = icp.plotPointsK(P2, Q, Pk2, Qk, "t2", "t1")
plt.title('ICP pose matching')
plt.text(-.2, 1.2, "Displacement: {} m, {} m".format(T[0],T[1]), fontsize=12)
plt.text(-.2, 1.1, "Angle Change: {} deg".format(angle), fontsize=12)

plt.show()
