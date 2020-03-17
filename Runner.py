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
        normalized_predictions = self.linear_regression_learner.predict(feature_data)
        predictions = np.around(DataService.min_max_denormalize_predictions(data[:, -1], normalized_predictions))

        PlotService.plot_line(
            x=range(1, len(self.linear_regression_learner.cost_history) + 1),
            y=self.linear_regression_learner.cost_history,
            x_label="Iteration",
            y_label="Training Cost",
            title="Training Learning Curve")

        x, y, z = self.build_model_plot_data(data, predictions)
        PlotService.plot3d_line(x, y, z, labels=['Age', 'Weight', 'BP'],
                                title="Linear Model: Blood Pressure for Age and Weight." )

        projected_data = data.copy()
        projected_data[:, -1] = predictions
        PlotService.plot3d_scatter_compare(data, projected_data, labels=['Age', 'Weight', 'BP'],
                                title="Actual vs Projected")

        labels = normalized_data[:, -1].flatten()
        cost = self.linear_regression_learner.calculate_cost(normalized_predictions, labels)
        print("Final Normalized Cost: ", cost)

        self.test(data)

    @staticmethod
    def build_model_plot_data(data, predictions):

        age = data[:, 0].flatten()
        weight = data[:, 1].flatten()
        min_predict_index = np.argmin(predictions)
        max_predict_index = np.argmax(predictions)
        x = [age[min_predict_index], age[max_predict_index]]
        y = [weight[min_predict_index], weight[max_predict_index]]
        z = [predictions[min_predict_index], predictions[max_predict_index]]
        return x, y, z

    def test(self, data):

        x_mins = np.amin(data, axis=0)
        x_maxs = np.amax(data, axis=0)

        min_age = x_mins[0]
        max_age = x_maxs[0]

        min_weight = x_mins[1]
        max_weight = x_maxs[1]

        min_blood_pressure = x_mins[2]
        max_blood_pressure = x_maxs[2]

        while True:

            age = input("Enter Age or type quit to exit: ")
            if age == 'quit':
                break
            weight = input("Enter Weight or type quit to exit: ")
            if weight == 'quit':
                break

            try:
                age = int(age)
                weight = int(weight)
            except ValueError:
                print("Age and Weight should be integers.", age, weight)
                break

            test_age = DataService.min_max_normalize_single(age, min_age, max_age)
            test_weight = DataService.min_max_normalize_single(weight, min_weight, max_weight)

            current_theta_normalized_projection = \
                self.linear_regression_learner.predict(np.array([test_age, test_weight]).reshape(1, 2))

            print("Current Theta Projection: ",
                  DataService.min_max_denormalize_single(current_theta_normalized_projection, min_blood_pressure,
                                                         max_blood_pressure))
            # best documented theta: -0.04973713  0.7198039   0.33875262
            best_theta = np.array([-0.04973713,  0.7198039,   0.33875262]).reshape((1, 3))
            best_theta_normalized_projection = self.linear_regression_learner\
                .predict_for_theta(np.array([test_age, test_weight]).reshape(1, 2), best_theta)

            print("Best Theta Projection: ",
                  DataService.min_max_denormalize_single(best_theta_normalized_projection, min_blood_pressure,
                                                         max_blood_pressure))



if __name__ == "__main__":

    Runner().run()
