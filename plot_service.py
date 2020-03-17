from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt


class PlotService:

    def __init__(self):
        pass

    @staticmethod
    def plot3d_scatter(data, labels, marker='o', title=''):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker=marker)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

        plt.title(title)

        plt.show()
