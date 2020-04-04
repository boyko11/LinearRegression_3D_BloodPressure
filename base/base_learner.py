from abc import ABC, abstractmethod


class BaseLearner(ABC):

    @abstractmethod
    def predict(self, feature_data):
        pass

    @abstractmethod
    def calculate_cost(self, predictions, labels):
        pass

    @abstractmethod
    def train(self, feature_data, labels):
        pass

