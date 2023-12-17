from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV


def find_best_regression_model(X_train, Y_train, cv=5):
    """
    使用交叉验证评估不同回归模型并返回最优模型。

    参数:
        X_train: 训练数据集输入特征，形状为 (n_samples, n_features) 的NumPy数组。
        Y_train: 训练数据集标签，形状为 (n_samples,) 的NumPy数组。
        cv: 交叉验证折数，默认为5。

    返回:
        best_model: 最优回归模型对象。
    """

    models = [
        LinearRegression(),
        KNeighborsRegressor(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        Ridge(),
        SVR()
    ]

    param_grid = {
        "LinearRegression": {},
        "KNeighborsRegressor": {"n_neighbors": [3, 5, 7]},
        "DecisionTreeRegressor": {"max_depth": [None, 1, 3, 5]},
        "RandomForestRegressor": {"n_estimators": [50, 100, 200]},
        "GradientBoostingRegressor": {"n_estimators": [50, 100, 200], "learning_rate": [0.5, 0.1, 0.05, 0.01]},
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "SVR": {"C": [0.1, 1.0, 10.0, 100], "kernel": ["linear", "rbf"]}
    }

    best_model = None
    best_score = float('inf')

    for model in models:
        model_name = type(model).__name__
        param_grid_model = param_grid[model_name]
        grid_search = GridSearchCV(model, param_grid_model, cv=cv, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, Y_train)
        mean_score = -np.mean(grid_search.cv_results_['mean_test_score'])

        if mean_score < best_score:
            best_score = mean_score
            best_model = grid_search.best_estimator_

    print("Best model:", best_model)
    print("Best score:", best_score)

    return best_model
