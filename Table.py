import numpy as np

class Table:
    poles = np.array([[-2, -2],
                    [2, -2],
                    [2, 2],
                    [-2, 2]])

    walls = np.array([[-1, -1],
                    [1, -1],
                    [1, 1],
                    [-1, 1]])

    def __init__(self):
        self.data = []

    def plotPoles(self, ax):
        ax.plot(self.poles[:,0] ,self.poles[:,1], 'bo');

    def plotWalls(self, ax):
        ax.plot(np.append(self.walls[:,0], self.walls[0,0]), np.append(self.walls[:,1], self.walls[0,1]), 'b-');
