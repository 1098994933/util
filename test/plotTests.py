import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from plot import plot_decision_tree
import matplotlib.pyplot as plt
import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot import plot_grouped_contour


class TestPlotFunctions(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        np.random.seed(42)
        n_points = 100

        # 创建两个组的数据
        self.x1 = np.random.uniform(0, 10, n_points)
        self.y1 = np.random.uniform(0, 10, n_points)
        self.z1 = self.x1 + self.y1 + np.random.normal(0, 0.5, n_points)

        self.x2 = np.random.uniform(0, 10, n_points)
        self.y2 = np.random.uniform(0, 10, n_points)
        self.z2 = self.x2 * self.y2 + np.random.normal(0, 0.5, n_points)

        # 合并数据
        self.x = np.concatenate([self.x1, self.x2])
        self.y = np.concatenate([self.y1, self.y2])
        self.z = np.concatenate([self.z1, self.z2])
        self.groups = np.concatenate([np.ones(n_points), np.ones(n_points) * 2])

    def test_plot_grouped_contour_basic(self):
        """测试基本的plot_grouped_contour功能"""
        fig, axes = plot_grouped_contour(
            self.x, self.y, self.z,
            groups=self.groups,
            resolution=20,
            levels=10
        )
        # 验证返回的图形对象
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_grouped_contour_with_constant_values(self):
        """测试包含常数值的情况"""
        # 创建一个组的数据为常数值，但添加一些小的随机扰动
        x_const = np.ones(50) * 5
        y_const = np.ones(50) * 5 + np.random.normal(0, 0.1, 50)
        z_const = np.ones(50) * 5 + np.random.normal(0, 0.1, 50)

        x = np.concatenate([x_const, self.x2])
        y = np.concatenate([y_const, self.y2])
        z = np.concatenate([z_const, self.z2])
        groups = np.concatenate([np.ones(50), np.ones(100) * 2])

        fig, axes = plot_grouped_contour(
            x, y, z,
            groups=groups,
            resolution=20,
            levels=10
        )
        plt.show()
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_grouped_contour_with_pandas_series(self):
        """测试使用pandas Series作为输入"""
        x_series = pd.Series(self.x, name='X轴')
        y_series = pd.Series(self.y, name='Y轴')
        z_series = pd.Series(self.z, name='Z值')

        fig, axes = plot_grouped_contour(
            x_series, y_series, z_series,
            groups=self.groups,
            resolution=20,
            levels=10
        )

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_grouped_contour_without_groups(self):
        """测试不提供groups参数的情况"""
        fig, axes = plot_grouped_contour(
            self.x, self.y, self.z,
            resolution=20,
            levels=10
        )

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_grouped_contour_with_custom_parameters(self):
        """测试自定义参数"""
        fig, axes = plot_grouped_contour(
            self.x, self.y, self.z,
            groups=self.groups,
            resolution=30,
            levels=15,
            cmap="viridis",
            figsize=(12, 8),
            title="自定义标题",
            share_colorbar=False
        )

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
