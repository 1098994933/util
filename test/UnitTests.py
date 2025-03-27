import unittest

import pandas as pd
from base_function import generate_alloys_random
from base_function import is_pareto_efficient
import numpy as np

from eval import cal_reg_metric
from plot import plot_regression_results

class MlTest(unittest.TestCase):
    def test_cal_metric(self):
        y_true = [1, 2, 3, 4]
        y_predict = [1, 2, 3, 4]
        metric = cal_reg_metric(y_true, y_predict)
        assert metric['MSE'] == 0
        assert metric['R2'] == 1
        print(metric)

    def test_true_predict_plot(self):
        y_true = [1, 2, 3, 4]
        y_predict = [1, 2, 3, 4]
        y_train = [-1, -2, -3, -4]
        y_train_predict = [-1, -2, -3, -4]
        plot_regression_results(y_true, y_predict, y_train, y_train_predict)

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
            "Ag": [80, 90],
            "condition1": [1, 2, 3, 4]
        }
        df = generate_alloys_random(search_space, residual_element="Zn", category_col=["condition1"], random_state=0)
        print(df)

    def test_is_pareto_efficient(self):
        data = np.array([[1, 1], [2, 2], [3, 3], [-1, 5]])
        is_pareto = is_pareto_efficient(data, return_mask=False)
        print(is_pareto)


class AutoDesignTests(unittest.TestCase):
    """
    测试自动材料设计
    """

    def setUp(self):
        self.df = pd.DataFrame({
            'col1': [1, 2, 6, 4] * 300,
            'col2': ['a', 'b', 'a', 'c'] * 300,
            'col3': [2, 2, 0, 4] * 300,
            "y": [3, 4, 6, 8] * 300
        })
        self.x_cols = ['col1', 'col2', 'col3']
        self.y_cols = ["y"]
        self.task_id = "abc"
        self.targets = [4]

    def test_fit(self):
        for y_col, target in zip(self.y_cols, self.targets):
            from DataPreprocessor import DataPreprocessor
            preprocessor = DataPreprocessor()
            preprocessor.fit(self.df)
            df_preprocessed = preprocessor.transform(self.df)
            print(df_preprocessed)
            # 取两者交集为x
            x_cols = [i for i in self.x_cols if i in df_preprocessed.columns]
            x = df_preprocessed[x_cols]
            y = df_preprocessed[y_col]
            print(x)
            from ml.FeatureSelection import FeatureSelector
            fs = FeatureSelector(x, y, select_method="auto")
            selected_features = fs.select_features_by_auto()
            fs.save_result(f"{self.task_id}__fs.pkl")
            x = x[selected_features]
            from ml.Regression import RegressionModel
            model = RegressionModel(x, y)
            best_model, best_model_info = model.fit_best_reg_model(self.task_id)
            print(best_model)
            print(best_model_info)
            from projection.PCA import PCAProjection
            n_components = min(len(x.columns), 2)
            projector = PCAProjection(n_components=n_components, n_clusters=3)
            projector.fit(x)
            X_pca = projector.transform(x)
            # 计算每个样本与设计目标的差距，选择差距小的为优类，否则为劣类
            y_pca = np.array(-1 * abs(y - target))
            from projection.OptimalProjection import OptimalProjection
            # op默认大为优类
            if n_components >= 2:
                op = OptimalProjection([X_pca], y_pca)
                op.find_best_projection()
                op.plot_decision_boundary()
                equations = op.get_rectangle_equations(pca=projector.pca, feature_names=x_cols)
                print(equations)
            else:
                print("n_components<2, cannot use OptimalProjection")

            # generate data by wae
            from generator.TableDataGenerator import TableDataGenerator
            print(self.df)
            generator = TableDataGenerator(self.df)
            generator.train(epochs=200)
            generate_df = generator.generate(num_samples=100)
            print("generate_df", generate_df)
            generate_df_preprocessed = preprocessor.transform(generate_df)
            generate_df[y_col] = best_model.predict(generate_df_preprocessed[selected_features])
            print(generate_df.head(50))


if __name__ == '__main__':
    unittest.main()
