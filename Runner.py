from data_service import DataService
from plot_service import PlotService
from linear_regression_learner import LinearRegressionLearner
import numpy as np


class Runner:

    def __init__(self):
        self.linear_regression_learner = None

    def run(self):
        # Load Data
        data = DataService.load_csv("data/blood_pressure_cengage.csv")
        self.linear_regression_learner = LinearRegressionLearner(data, learning_rate=0.001)

        # PlotService.plot3d_scatter(data, labels=['Age', 'Weight', 'BP'], title="Blood Pressure for Age and Weight.")

        normalized_data = DataService.min_max_normalize(data)

        # PlotService.plot3d_scatter(normalized_data, labels=['Age', 'Weight', 'BP'],
        #                            title="BP for Age and Weight. Min-Max normalized.")

        self.linear_regression_learner.train(normalized_data)

        feature_data = normalized_data[:, :-1]
        labels = normalized_data[:, -1].flatten()
        predictions = self.linear_regression_learner.predict_all_records(feature_data)
        cost = self.linear_regression_learner.calculate_cost(predictions, labels)

        print("Labels: ", data[:, -1])
        print("Predictions", np.around(DataService.min_max_denormalize_predictions(data[:, -1], predictions)))
        print('Cost: ', cost)

        PlotService.plot_line(
            x=range(1, len(self.linear_regression_learner.cost_history) + 1),
            y=self.linear_regression_learner.cost_history,
            x_label="Iteration",
            y_label="Training Cost",
            title="Training Learning Curve")



if __name__ == "__main__":

    Runner().run()
