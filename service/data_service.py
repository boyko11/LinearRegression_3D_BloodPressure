import numpy as np


class DataService:

    def __init__(self):
        np.set_printoptions(suppress=True)

    @staticmethod
    def load_csv(file_name):
        return np.loadtxt(open(file_name, "rb"), delimiter=",")

    @staticmethod
    def normalize(data, method='min-max'):

        if method == 'min-max':
            return DataService.min_max_normalize(data)

        return DataService.zscore_normalize(data)

    @staticmethod
    def min_max_normalize(data):

        np.set_printoptions(suppress=True)
        x_mins = np.amin(data, axis=0).reshape(1, data.shape[1])
        x_mins = np.repeat(x_mins, data.shape[0], axis=0)

        x_maxs = np.amax(data, axis=0).reshape(1, data.shape[1])
        x_maxs = np.repeat(x_maxs, data.shape[0], axis=0)

        return (data - x_mins) / (x_maxs - x_mins)

    @staticmethod
    def zscore_normalize(data):

        means = np.mean(data, axis=0).reshape(1, data.shape[1])
        means = np.repeat(means, data.shape[0], axis=0)
        stds = np.std(data, axis=0).reshape(1, data.shape[1])
        stds = np.repeat(stds, data.shape[0], axis=0)
        return (data - means) / stds

    @staticmethod
    def denormalize_predictions(actual_labels, predictions, method='min-max'):

        if method == 'min-max':
            return DataService.min_max_denormalize_predictions(actual_labels, predictions)

        return DataService.zscore_denormalize_predictions(actual_labels, predictions)

    @staticmethod
    def min_max_denormalize_predictions(actual_labels, predictions):

        np.set_printoptions(suppress=True)
        labels_min = np.amin(actual_labels)
        labels_max = np.amax(actual_labels)

        return predictions * (labels_max - labels_min) + labels_min

    @staticmethod
    def zscore_denormalize_predictions(actual_labels, predictions):

        np.set_printoptions(suppress=True)
        labels_mean = np.mean(actual_labels)
        labels_std = np.std(actual_labels)

        return predictions * labels_std + labels_mean

    @staticmethod
    def normalize_single(number_to_normalize, method='min-max', min=None, max=None, mean=None, std=None):

        if method == 'min-max':
            return (number_to_normalize - min) / (max - min)

        return (number_to_normalize - mean) / std

    @staticmethod
    def denormalize_single(number_to_denormalize, method='min-max', min=None, max=None, mean=None, std=None):

        if method == 'min-max':
            return number_to_denormalize * (max - min) + min

        # z-score denormalization
        return number_to_denormalize * std + mean



