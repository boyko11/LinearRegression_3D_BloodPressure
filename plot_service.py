from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d


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

    @staticmethod
    def plot3d_scatter_compare(feature_data, labels_data, predictions, labels=['', '', ''], marker1='o', marker2='x', color1=None, color2=None, title=''):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(feature_data[:, 0], feature_data[:, 1], labels_data, marker=marker1, label="Actual")
        ax.scatter(feature_data[:, 0], feature_data[:, 1], predictions, marker=marker2, label="Projected")

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.legend()

        plt.title(title)

        plt.show()

    def plot3d_line(x, y, z, labels, title=''):

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(x, y, zs=z, label='Linear Model')

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

        plt.title(title)

        plt.show()

    @staticmethod
    def plot_line(x, y, x_label, y_label, title=''):

        plt.figure()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.grid()

        plt.plot(x, y)

        plt.show()
