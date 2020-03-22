from data_service import DataService
from plot_service import PlotService
from linear_regression_learner import LinearRegressionLearner
import numpy as np


class Runner:

    def __init__(self, normalization_method='min-max'):
        self.linear_regression_learner = None
        self.normalization_method = normalization_method

    def run(self):
        # Load Data
        data = DataService.load_csv("data/blood_pressure_cengage.csv")
        self.linear_regression_learner = LinearRegressionLearner(data, learning_rate=0.001)

        # PlotService.plot3d_scatter(data, labels=['Age', 'Weight', 'BP'], title="Blood Pressure for Age and Weight.")

        normalized_data = DataService.normalize(data, method='min-max')

        # PlotService.plot3d_scatter(normalized_data, labels=['Age', 'Weight', 'BP'],
        #                            title="BP for Age and Weight. Min-Max normalized.")

        self.linear_regression_learner.train(normalized_data)

        feature_data = normalized_data[:, :-1]
        normalized_predictions = self.linear_regression_learner.predict(feature_data)
        predictions = np.around(DataService.denormalize_predictions(data[:, -1], normalized_predictions,
                                                                    method='min-max'))

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

        # self.test(data)

        print("Normal Equation: ")

        # Normal equation
        feature_data_bias = np.insert(feature_data, 0, 1, axis=1)
        x_trans_x = np.dot(np.transpose(feature_data_bias), feature_data_bias)
        norm_equation_theta = np.dot(np.dot(np.linalg.inv(x_trans_x), np.transpose(feature_data_bias)), labels)
        print(norm_equation_theta)

        self.linear_regression_learner.theta = norm_equation_theta

        normalized_predictions = self.linear_regression_learner.predict(feature_data)
        predictions = np.around(DataService.denormalize_predictions(data[:, -1], normalized_predictions,
                                                                    method='min-max'))
        cost = self.linear_regression_learner.calculate_cost(normalized_predictions, labels)
        print("Final Norm Equation Normalized Cost: ", cost)

        x, y, z = self.build_model_plot_data(data, predictions)
        PlotService.plot3d_line(x, y, z, labels=['Age', 'Weight', 'BP'],
                                title="Normal Equation Model: Blood Pressure for Age and Weight." )

        projected_data = data.copy()
        projected_data[:, -1] = predictions
        PlotService.plot3d_scatter_compare(data, projected_data, labels=['Age', 'Weight', 'BP'],
                                title="Normal Equation Actual vs Projected")


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

        x_means = np.mean(data, axis=0)
        mean_age = x_means[0]
        mean_weight = x_means[1]
        mean_blood_pressure = x_means[2]

        x_stds = np.std(data, axis=0)
        std_age = x_stds[0]
        std_weight = x_stds[1]
        std_blood_pressure = x_stds[2]

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

            test_age = DataService.normalize_single(age, min=min_age, max=max_age, mean=mean_age, std=std_age,
                                                    method=self.normalization_method)
            test_weight = DataService.normalize_single(weight, min=min_weight, max=max_weight, mean=mean_weight,
                                                       std=std_weight, method=self.normalization_method)

            current_theta_normalized_projection = \
                self.linear_regression_learner.predict(np.array([test_age, test_weight]).reshape(1, 2))

            print("Current Theta Projection: ",
                  DataService.denormalize_single(current_theta_normalized_projection, method=self.normalization_method,
                                                 min=min_blood_pressure, max=max_blood_pressure,
                                                 mean=mean_blood_pressure, std=std_blood_pressure))
            # best documented theta min-max norm: -0.04973713  0.7198039   0.33875262
            # best documented theta zscore norm: -0.03447741  0.70838301  0.32258052
            best_theta = np.array([-0.03447741, 0.70838301, 0.32258052]).reshape((1, 3))
            if self.normalization_method == 'min-max':
                best_theta = np.array([-0.04973713,  0.7198039,   0.33875262]).reshape((1, 3))

            best_theta_normalized_projection = self.linear_regression_learner\
                .predict_for_theta(np.array([test_age, test_weight]).reshape(1, 2), best_theta)

            print("Best Theta Projection: ",
                  DataService.denormalize_single(best_theta_normalized_projection, method=self.normalization_method,
                                                 min=min_blood_pressure, max=max_blood_pressure,
                                                 mean=mean_blood_pressure, std=std_blood_pressure))


if __name__ == "__main__":

    Runner(normalization_method='z').run()
