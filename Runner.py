from data_service import DataService
from plot_service import PlotService


class Runner:

    def __init__(self):
        pass

    @staticmethod
    def run():
        # Load Data
        data = DataService.load_csv("data/blood_pressure_cengage.csv")
        PlotService.plot3d_scatter(data, labels=['Age', 'Weight', 'BP'], title="Blood Pressure for Age and Weight.")


if __name__ == "__main__":

    Runner().run()
