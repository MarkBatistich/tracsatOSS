import numpy as np
import matplotlib.pyplot as plt
import sys
from math import sin, cos, atan2, pi

#This implements the functions required to run the iterative closest point algorithm
#Much guidance from: https://nbviewer.org/github/niosus/notebooks/blob/master/icp.ipynb

iterations = 5;

#Associate each point in P with the closest point in Q
#This is a brute force algorithm, running in O(n^2)
#Consider a better option for the future

def associatePoints(P, Q):
    pSize = len(P[0])
    qSize = len(Q[0])
    centroidP = np.array([np.mean(P, axis = 1)]).T
    centroidQ = np.array([np.mean(Q, axis = 1)]).T
    associations = []
    for i in range(pSize):
        minDistance = sys.maxsize
        minIndex = -1
        for j in range(qSize):
            pPoint = np.array([P[:, i]]).T - centroidP
            qPoint = np.array([Q[:, j]]).T - centroidQ
            distance = np.linalg.norm(pPoint - qPoint)
            if distance < minDistance:
                minDistance = distance
                minIndex = j
        associations.append((i, minIndex))
    #print(pPoint, centroidP)
    return associations

def plotPoints(data1, data2, label1, label2, markersize_1=8, markersize_2=8):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    if data1 is not None:
        x_p, y_p = data1
        ax.plot(x_p, y_p, color='#336699', markersize=markersize_1, marker='o', linestyle=":", label=label1)
    if data2 is not None:
        x_q, y_q = data2
        ax.plot(x_q, y_q, color='orangered', markersize=markersize_2, marker='o', linestyle=":", label=label2)
    ax.legend()
    return ax

def drawAssociations(P, Q, associations, ax):
    label_added = False
    for i, j in associations:
        x = [P[0, i], Q[0, j]]
        y = [P[1, i], Q[1, j]]
        if not label_added:
            ax.plot(x, y, color='grey', label='correpondences')
            label_added = True
        else:
            ax.plot(x, y, color='grey')
    ax.legend()

def findOptimalTransformation(P, Q, associations):
    centroidP = np.array([np.mean(P, axis = 1)]).T
    centroidQ = np.array([np.mean(Q, axis = 1)]).T
    spread = np.zeros((2, 2))
    for i, j in associations:
        pPoint = np.array([P[:, i]]).T
        qPoint = np.array([Q[:, j]]).T
        spread += np.dot((pPoint - centroidP), (qPoint - centroidQ).T)
        #print(np.matmul((pPoint - centroidP),(qPoint - centroidQ).transpose()))
    spread /= len(P[0]);
    U, S, VT = np.linalg.svd(spread)
    S_new = np.array([[1, 0], [0, np.linalg.det(U)*np.linalg.det(VT)]])
    rotation = U @ S_new @ VT
    translation = centroidQ - rotation.T @ centroidP
    #print(translation)
    return rotation, translation

def icp(P,Q):
    C = np.identity(2)
    T = np.zeros((2, 1))
    for i in range(iterations):
        associations = associatePoints(P,Q)
        rotation, translation = findOptimalTransformation(P, Q, associations)
        P = rotation.T @ P + translation
        T = rotation.T @ T
        C = C @ rotation
        T = T + translation
        prevTranslation = translation
    return C, T

def angleOfRot(C):
    return atan2(C[0,1], C[0,0])

def createPointList(rad, theta):
    x = rad*np.cos(theta)
    y = rad*np.sin(theta)
    points = np.vstack((x,y))
    return points


# initialize pertrubation rotation
# angle = pi / 4
# R_true = np.array([[cos(angle), -sin(angle)],
#                    [sin(angle),  cos(angle)]])
# t_true = np.array([[-2], [5]])
#
# # Generate data as a list of 2d points
# num_points = 30
# true_data = np.zeros((2, num_points))
# true_data[0, :] = range(0, num_points)
# true_data[1, :] = 0.2 * true_data[0, :] * np.sin(0.5 * true_data[0, :])
# # Move the data
# moved_data = R_true.dot(true_data) + t_true
#
# # Assign to variables we use in formulas.
# Q = true_data
# P = moved_data
#
# C,T = icp(P,Q)
# print(C)
# print(T)
#
# P2 = C.T @ P + T
# ax = plotPoints(P2, Q, label1='Pnew', label2='Q')
#
# # ax = plotPoints(P, Q, "P", "Q")
# # associations = associatePoints(P,Q)
# # drawAssociations(P, Q, associations, ax)
# # rotation, translation = findOptimalTransformation(P, Q, associations)
# # P2 = rotation.T @ P + translation
# # ax = plotPoints(P2, Q, label1='P2', label2='Q')
# # associations = associatePoints(P2,Q)
# # drawAssociations(P2, Q, associations, ax)
# # rotation, translation = findOptimalTransformation(P2, Q, associations)
# # P3 = rotation.T @ P2 + translation
# # ax = plotPoints(P3, Q, label1='P3', label2='Q')
# plt.show()
