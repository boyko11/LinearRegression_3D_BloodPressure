import numpy as np
from base.base_learner import BaseLearner


class LinearRegressionLearner(BaseLearner):

    def __init__(self, data, learning_rate=0.001):
        self.theta = np.random.rand(1, data.shape[1])
        self.learning_rate = learning_rate
        self.cost_history = []
        self.theta_history = []
        # this is a bit misleading, the number of thetas should be the number of features + 1
        # since data contains all features plus the label,
        # it just happens the number of columns is the same as the number of thetas

    def predict(self, feature_data):

        return np.dot(np.insert(feature_data, 0, 1, axis=1), np.transpose(self.theta)).flatten()

    @staticmethod
    def predict_for_theta(feature_data, theta):

        return np.dot(np.insert(feature_data, 0, 1, axis=1), np.transpose(theta)).flatten()

    def calculate_cost(self, predictions, labels):

        return np.sum(np.square(labels - predictions)) / 2 * predictions.shape[0]

    def train(self, feature_data, labels):

        for i in range(15000):
            predictions = self.predict(feature_data)
            current_cost = self.calculate_cost(predictions, labels)
            self.cost_history.append(current_cost)
            self.theta_history.append(self.theta)
            self.update_theta_gradient_descent(predictions, feature_data, labels)

        min_cost_index = np.argmin(self.cost_history)
        self.theta = self.theta_history[min_cost_index]

        print('min_cost_index: ', min_cost_index)
        print('min_cost_theta: ', self.theta)

        self.cost_history = self.cost_history[:min_cost_index + 1]

    def update_theta_gradient_descent(self, predictions, feature_data, labels):

        predictions_minus_labels = np.transpose(predictions - labels)

        predictions_minus_labels = predictions_minus_labels.reshape(predictions_minus_labels.shape[0], 1)

        gradient = np.mean(predictions_minus_labels * feature_data, axis=0)
        #add 1 for the bias
        gradient = np.concatenate(([1], gradient))

        self.theta = self.theta - self.learning_rate * gradient


