import numpy as np

class DataService:

    def __init__(self):
        np.set_printoptions(suppress=True)

    @staticmethod
    def load_csv(file_name):
        return np.loadtxt(open(file_name, "rb"), delimiter=",")

    @staticmethod
    def min_max_normalize(data):

        np.set_printoptions(suppress=True)
        x_mins = np.amin(data, axis=0).reshape(1, data.shape[1])
        x_mins = np.repeat(x_mins, data.shape[0], axis=0)

        x_maxs = np.amax(data, axis=0).reshape(1, data.shape[1])
        x_maxs = np.repeat(x_maxs, data.shape[0], axis=0)

        return (data - x_mins) / (x_maxs - x_mins)
