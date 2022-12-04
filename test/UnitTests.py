import unittest
from ml.vis import cal_metric, true_predict_plot


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


if __name__ == '__main__':
    unittest.main()
