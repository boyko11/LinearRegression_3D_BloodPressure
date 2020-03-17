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
        predictions = np.around(DataService.min_max_denormalize_predictions(data[:, -1], predictions))
        print("Predictions", )
        print('Cost: ', cost)

        PlotService.plot_line(
            x=range(1, len(self.linear_regression_learner.cost_history) + 1),
            y=self.linear_regression_learner.cost_history,
            x_label="Iteration",
            y_label="Training Cost",
            title="Training Learning Curve")

        #best documented theta: -0.03028008  0.52666306  0.51706439

        age = data[:, 0].flatten()
        weight = data[:, 1].flatten()
        min_predict_index = np.argmin(predictions)
        max_predict_index = np.argmax(predictions)
        x = [age[min_predict_index], age[max_predict_index]]
        y = [weight[min_predict_index], weight[max_predict_index]]
        z = [predictions[min_predict_index], predictions[max_predict_index]]

        PlotService.plot3d_line(x, y, z, labels=['Age', 'Weight', 'BP'],
                                title="Linear Model: Blood Pressure for Age and Weight." )

        projected = data.copy()
        projected[:, -1] = predictions
        PlotService.plot3d_scatter_compare(data, projected, labels=['Age', 'Weight', 'BP'],
                                title="Actual vs Projected")


if __name__ == "__main__":

    Runner().run()
