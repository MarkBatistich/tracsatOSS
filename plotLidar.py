import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

file = open('data/angle_0deg_0.csv')
csvreader = csv.reader(file)
th = []
for row in csvreader:
    th.append(row)
file.close()
#print(th)

data = pd.read_csv('data/angle_0deg_1.csv', header=None)
data = data[0].astype(float)
theta = data.to_numpy()

data = pd.read_csv('data/distance_0deg_1.csv', header=None)
data = data[0].astype(float)
rad = data.to_numpy()

theta = (np.linspace(-45,225,811)*np.pi/180.0).tolist()

plt.figure()
plt.polar(theta,rad,'b.')
plt.show()
