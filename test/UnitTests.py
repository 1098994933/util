import unittest
from ml.vis import cal_metric, true_predict_plot
from base_function import generate_alloys_random


class MlTest(unittest.TestCase):
    def test_cal_metric(self):
        y_true = [1, 2, 3, 4]
        y_predict = [1, 2, 3, 4]
        metric = cal_metric(y_true, y_predict)
        assert metric['MSE'] == 0
        assert metric['R2'] == 1
        print(metric)

    def test_true_predict_plot(self):
        y_true = [1, 2, 3, 4]
        y_predict = [1, 2, 3, 4]
        y_train = [-1, -2, -3, -4]
        y_train_predict = [-1, -2, -3, -4]
        true_predict_plot(y_true, y_predict, y_train, y_train_predict)

    def test_generate_alloys_random(self):
        search_space = {
            "Al": [0, 30],
            "Cu": [1, 2]
        }
        df = generate_alloys_random(search_space, residual_element="Zn")
        print(df)

        search_space = {
            "Al": [0, 30],
            "Cu": [1, 2],
            "condition1": [1, 2, 3, 4]
        }
        df = generate_alloys_random(search_space, residual_element="Zn", category_col=["condition1"],random_state=0)
        print(df)


if __name__ == '__main__':
    unittest.main()
