import numpy as np

class DataService:

    def __init__(self):
        pass

    @staticmethod
    def load_csv(file_name):
        return np.loadtxt(open(file_name, "rb"), delimiter=",")