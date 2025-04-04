"""
Multiple Objective Optimization for Machine Learning Models
"""
import unittest
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from sklearn.ensemble import RandomForestRegressor


class MLModelProblem(Problem):
    def __init__(self, models, features_list, objectives_directions, x_bounds):
        """
        params:
            models: list of trained ML models
            features_list: list of feature index lists for each model
            objectives_directions: list of 'min'/'max' for each objective
            x_bounds: list of (min, max) tuples for each design variable
        """
        self.models = models
        self.features_list = [np.array(f) for f in features_list]
        self.directions = objectives_directions

        # 自动推断变量维度
        all_features = np.unique(np.concatenate(self.features_list))
        n_var = len(all_features)

        # 设置变量边界
        xl = np.zeros(n_var)
        xu = np.zeros(n_var)
        for idx in all_features:
            xl[idx], xu[idx] = x_bounds[idx]

        super().__init__(n_var=n_var,
                         n_obj=len(models),
                         xl=xl,
                         xu=xu,
                         type_var=np.float64)

    def _evaluate(self, x, out, *args, **kwargs):
        F = []
        for i, (model, features) in enumerate(zip(self.models, self.features_list)):
            # 提取每个模型的特征子集
            X_sub = x[:, features]

            # 获取预测结果并确保形状正确
            y_predict = model.predict(X_sub).reshape(-1, 1)

            # 根据优化方向调整目标值
            if self.directions[i] == 'max':
                y_predict = -y_predict  # pymoo默认最小化，最大化目标取负数
            F.append(y_predict)

        out["F"] = np.hstack(F)


class MultipleModelOptimizer:
    def __init__(self, models, features_list: List[List], objectives_directions: List[str], x_bounds):
        """
        params:
            models: list of pre-trained ML models
            features_list: list of lists containing feature indices for each model
            objectives_directions: list of 'min'/'max' for each model's objective
            x_bounds: list of tuples specifying (min, max) for each unique feature
        """
        # validate input parameters.
        self.validate_input(models, features_list, objectives_directions, x_bounds)

        self.models = models
        self.features_list = features_list
        self.directions = objectives_directions
        self.x_bounds = x_bounds

        # 创建优化问题
        self.problem = MLModelProblem(models,
                                      features_list,
                                      objectives_directions,
                                      x_bounds)

        # 算法参数
        self.algorithm = NSGA2(pop_size=100)
        self.termination = ('n_gen', 100)

    @staticmethod
    def validate_input(models, features_list, directions, x_bounds):
        assert len(models) == len(features_list) == len(directions), \
            "Models, features and directions must have same length"

        all_features = set()
        for fl in features_list:
            all_features.update(fl)

        max_feature = max(all_features) if all_features else 0
        assert max_feature < len(x_bounds), \
            "Feature index exceeds specified variable bounds"

        valid_directions = {'min', 'max'}
        assert all(d in valid_directions for d in directions), \
            "Invalid optimization direction. Use 'min' or 'max'"

    def optimize(self, verbose=True):
        res = minimize(
            self.problem,
            self.algorithm,
            self.termination,
            seed=42,
            verbose=verbose,
            save_history=True
        )
        return res

    def set_algorithm_params(self, pop_size=100, n_gen=100):
        self.algorithm.pop_size = pop_size
        self.termination = ('n_gen', n_gen)

    def get_result_df(self, res, feature_names=None, y_names=None):
        """
        将优化结果转换为包含特征值和目标值的DataFrame

        参数:
            res: pymoo优化结果对象
            feature_names: 特征名称列表，默认使用x0,x1...
            y_names: 目标名称列表，默认使用y1,y2...

        返回:
            pandas DataFrame 包含特征值和调整后的目标值
        """
        # 处理特征名称
        if feature_names is None:
            feature_names = [f'x{i}' for i in range(self.problem.n_var)]
        else:
            if len(feature_names) != self.problem.n_var:
                raise ValueError("特征名称数量与变量维度不一致")

        # 处理目标名称
        n_obj = self.problem.n_obj
        if y_names is None:
            y_names = [f'y{i + 1}' for i in range(n_obj)]
        else:
            if len(y_names) != n_obj:
                raise ValueError("目标名称数量与目标维度不一致")

        # 调整目标值符号（还原原始预测值）
        adjusted_F = res.F.copy()
        for i, direction in enumerate(self.directions):
            if direction == 'max':
                adjusted_F[:, i] = -adjusted_F[:, i]

        # 创建DataFrame
        df_x = pd.DataFrame(res.X, columns=feature_names)
        df_y = pd.DataFrame(adjusted_F, columns=y_names)
        return pd.concat([df_y, df_x], axis=1)


class MultipleObjectiveOptimizationTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_moo(self):
        # 1. 模拟训练数据（假设原始数据范围）
        np.random.seed(42)
        X1 = np.random.uniform(0, 10, 100).reshape(-1, 1)  # 共享特征
        X2a = np.random.uniform(0, 5, 100).reshape(-1, 1)  # 模型1独特征
        X2b = np.random.uniform(-5, 0, 100).reshape(-1, 1)  # 模型2独特征
        y1 = 2 * X1 + 3 * X2a  # 线性模型
        y2 = 1/X1 ** 2 + 4 * X2b  # 二次模型

        model1 = RandomForestRegressor()
        model1.fit(np.hstack([X1, X2a]), y1)

        model2 = RandomForestRegressor()
        model2.fit(np.hstack([X1, X2b]), y2)

        # 假设有两个预训练模型和以下参数
        models = [model1, model2]
        features_list = [
            [0, 1],  # 模型1使用特征0和1
            [0, 2]  # 模型2使用特征0和2
        ]
        objectives_directions = ['max', 'max']  # 最大化模型1输出，最小化模型2输出
        x_bounds = [  # 三个设计变量的边界（特征0,1,2）
            (0, 1),  # 特征0的范围
            (-1, 1),  # 特征1的范围
            (0, 10)  # 特征2的范围
        ]

        # 创建优化器
        optimizer = MultipleModelOptimizer(models,
                                           features_list,
                                           objectives_directions,
                                           x_bounds)

        # 设置优化参数
        optimizer.set_algorithm_params(pop_size=50, n_gen=200)

        # 执行优化
        result = optimizer.optimize()

        # 获取帕累托前沿解
        pareto_solutions = result.X
        pareto_objectives = result.F
        print("Pareto Front Solutions:")
        print(pareto_solutions)
        print("Pareto Front Objectives:")
        print(pareto_objectives)

        # 测试自定义列名
        custom_features = ['temperature', 'pressure', 'humidity']
        custom_targets = ['efficiency', 'cost']
        df_custom = optimizer.get_result_df(
            result,
            feature_names=custom_features,
            y_names=custom_targets
        )
        print(df_custom)
        plt.scatter(df_custom['cost'], df_custom['efficiency'])
        plt.show()
