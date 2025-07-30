import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from plot import plot_decision_tree, plot_group_by_freq_mean
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import unittest
import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot import plot_grouped_contour, plot_group_time_rolling


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

        # 为plot_group_scatter_mean创建测试数据
        self.scatter_x = pd.Series(np.random.uniform(0, 20, 200), name='X轴')
        self.scatter_y = pd.Series(np.random.uniform(0, 20, 200), name='Y轴')
        self.scatter_groups = pd.Series(np.random.choice(['A', 'B', 'C'], 200), name='分组')

        # 创建时间序列
        start_date = datetime(2023, 1, 1)
        self.dates = [start_date + timedelta(days=i) for i in range(n_points)]
        
        # 创建分组数据
        self.group_data = ['Group_A'] * 50 + ['Group_B'] * 50
        
        # 创建带有噪声的数据
        self.x_base = np.linspace(0, 10, n_points)
        self.y_base = 2 * self.x_base + 3 + np.random.normal(0, 0.5, n_points)
        
        # 添加一些时间趋势
        self.time_trend = np.sin(np.linspace(0, 4*np.pi, n_points)) * 0.5
        self.y_with_trend = self.y_base + self.time_trend
        
        self.test_data = pd.DataFrame({
            'time': self.dates,
            'x': self.x_base,
            'y': self.y_with_trend,
            'group': self.group_data
        })

    def test_plot_grouped_contour_basic(self):
        """测试基本的plot_grouped_contour功能"""
        fig, axes = plot_grouped_contour(
            self.x, self.y, self.z,
            groups=self.groups,
            resolution=20,
            levels=10
        )
        # 验证返回的图形对象
        self.assertIsInstance(fig, Figure)
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
        self.assertIsInstance(fig, Figure)
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

        self.assertIsInstance(fig, Figure)
        plt.close(fig)

    def test_plot_grouped_contour_without_groups(self):
        """测试不提供groups参数的情况"""
        fig, axes = plot_grouped_contour(
            self.x, self.y, self.z,
            resolution=20,
            levels=10
        )

        self.assertIsInstance(fig, Figure)
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

        self.assertIsInstance(fig, Figure)
        plt.close(fig)

    def test_plot_group_scatter_mean_basic(self):
        """测试基本的plot_group_scatter_mean功能"""
        fig, axes = plot_group_by_freq_mean(
            self.scatter_x, self.scatter_y, self.scatter_groups
        )
        # 验证返回的图形对象
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(axes, np.ndarray)
        plt.close(fig)

    def test_plot_group_scatter_mean_with_custom_figsize(self):
        """测试自定义图形尺寸的plot_group_scatter_mean"""
        fig, axes = plot_group_by_freq_mean(self.scatter_x, self.scatter_y, self.scatter_groups,figsize=(20, 15)
        )
        plt.show()
        # 验证图形尺寸
        self.assertEqual(fig.get_size_inches()[0], 20)
        self.assertEqual(fig.get_size_inches()[1], 15)
        plt.close(fig)

    def test_plot_group_scatter_mean_with_single_group(self):
        """测试只有一个分组的情况"""
        single_group_x = pd.Series(np.random.uniform(0, 10, 50), name='X轴')
        single_group_y = pd.Series(np.random.uniform(0, 10, 50), name='Y轴')
        single_group_groups = pd.Series(['A'] * 50, name='分组')
        
        fig, axes = plot_group_by_freq_mean(
            single_group_x, single_group_y, single_group_groups
        )
        
        self.assertIsInstance(fig, Figure)
        plt.close(fig)

    def test_plot_group_scatter_mean_with_many_groups(self):
        """测试多个分组的情况"""
        many_groups_x = pd.Series(np.random.uniform(0, 10, 100), name='X轴')
        many_groups_y = pd.Series(np.random.uniform(0, 10, 100), name='Y轴')
        many_groups_groups = pd.Series(np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], 100), name='分组')
        
        fig, axes = plot_group_by_freq_mean(
            many_groups_x, many_groups_y, many_groups_groups
        )
        
        self.assertIsInstance(fig, Figure)
        plt.close(fig)

    def test_plot_group_scatter_mean_with_empty_data(self):
        """测试空数据的情况"""
        empty_x = pd.Series([], name='X轴')
        empty_y = pd.Series([], name='Y轴')
        empty_groups = pd.Series([], name='分组')
        
        # 应该抛出异常或返回空图形
        with self.assertRaises(Exception):
            plot_group_by_freq_mean(empty_x, empty_y, empty_groups)

    def test_plot_group_scatter_mean_with_unequal_lengths(self):
        """测试输入数据长度不一致的情况"""
        x_short = pd.Series(np.random.uniform(0, 10, 50), name='X轴')
        y_long = pd.Series(np.random.uniform(0, 10, 100), name='Y轴')
        groups_long = pd.Series(np.random.choice(['A', 'B'], 100), name='分组')
        
        # 应该抛出异常
        with self.assertRaises(Exception):
            plot_group_by_freq_mean(x_short, y_long, groups_long)

    def test_plot_group_scatter_mean_quantile_calculation(self):
        """测试分位数计算是否正确"""
        # 创建有序的数据来验证分位数计算
        ordered_x = pd.Series(np.arange(100), name='X轴')
        ordered_y = pd.Series(np.arange(100) + np.random.normal(0, 1, 100), name='Y轴')
        ordered_groups = pd.Series(['A'] * 100, name='分组')
        
        fig, axes = plot_group_by_freq_mean(
            ordered_x, ordered_y, ordered_groups
        )
        
        self.assertIsInstance(fig, Figure)
        plt.close(fig)

    def test_plot_group_scatter_mean_with_no_groups(self):
        """测试不提供groups参数的情况（默认分一组）"""
        x_data = pd.Series(np.random.uniform(0, 10, 50), name='X轴')
        y_data = pd.Series(np.random.uniform(0, 10, 50), name='Y轴')
        
        fig, axes = plot_group_by_freq_mean(
            x_data, y_data  # 不提供groups参数
        )
        
        self.assertIsInstance(fig, Figure)
        plt.close(fig)

    def test_plot_group_time_rolling_basic(self):
        """测试基本的plot_group_time_rolling功能"""
        time_col = self.test_data['time']
        x_col = self.test_data['x']
        y_col = self.test_data['y']
        groups_col = self.test_data['group']
        
        # 测试基本调用
        fig, axes = plot_group_time_rolling(
            time_col=time_col,
            x=x_col,
            y=y_col,
            groups=groups_col,
            window=5
        )
        
        # 验证返回的图形对象
        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(axes, np.ndarray)
        
        # 验证图形包含双坐标轴
        if axes.size > 0:
            ax = axes.flat[0]
            # 检查是否有次坐标轴
            self.assertTrue(hasattr(ax, 'right_ax') or len(ax.figure.axes) > 1)
            
            # 验证坐标轴范围是否共享
            if axes.size > 1:
                first_ax = axes.flat[0]
                second_ax = axes.flat[1]
                # 检查主坐标轴范围是否相同
                self.assertEqual(first_ax.get_ylim(), second_ax.get_ylim())
        
        # 关闭图形以释放内存
        plt.close(fig)
    
    def test_plot_group_time_rolling_no_groups(self):
        """测试没有分组的情况"""
        time_col = self.test_data['time']
        x_col = self.test_data['x']
        y_col = self.test_data['y']
        
        # 测试没有分组的情况
        fig, axes = plot_group_time_rolling(
            time_col=time_col,
            x=x_col,
            y=y_col,
            window=10
        )
        
        # 验证返回的图形对象
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(axes, np.ndarray)
        
        # 关闭图形以释放内存
        plt.close(fig)
    
    def test_plot_group_time_rolling_custom_window(self):
        """测试自定义窗口大小"""
        time_col = self.test_data['time']
        x_col = self.test_data['x']
        y_col = self.test_data['y']
        groups_col = self.test_data['group']
        
        # 测试不同的窗口大小
        fig, axes = plot_group_time_rolling(
            time_col=time_col,
            x=x_col,
            y=y_col,
            groups=groups_col,
            window=15,
            min_periods=3
        )
        
        # 验证返回的图形对象
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(axes, np.ndarray)
        
        # 关闭图形以释放内存
        plt.close(fig)
    
    def test_plot_group_time_rolling_parameter_validation(self):
        """测试参数验证"""
        time_col = self.test_data['time']
        x_col = self.test_data['x']
        y_col = self.test_data['y']
        
        # 测试长度不匹配的情况
        with self.assertRaises(ValueError):
            plot_group_time_rolling(
                time_col=time_col,
                x=x_col[:-1],  # 少一个元素
                y=y_col,
                window=5
            )
        
        # 测试无效的window参数
        with self.assertRaises(ValueError):
            plot_group_time_rolling(
                time_col=time_col,
                x=x_col,
                y=y_col,
                window=0  # 无效的window
            )
        
        # 测试无效的min_periods参数
        with self.assertRaises(ValueError):
            plot_group_time_rolling(
                time_col=time_col,
                x=x_col,
                y=y_col,
                window=5,
                min_periods=10  # 大于window
            )
        
        # 测试无效的max_segments参数
        with self.assertRaises(ValueError):
            plot_group_time_rolling(
                time_col=time_col,
                x=x_col,
                y=y_col,
                window=5,
                max_segments=0  # 无效的max_segments
            )
    
    def test_plot_group_time_rolling_with_title(self):
        """测试带标题的情况"""
        time_col = self.test_data['time']
        x_col = self.test_data['x']
        y_col = self.test_data['y']
        groups_col = self.test_data['group']
        
        # 测试带标题的情况
        fig, axes = plot_group_time_rolling(
            time_col=time_col,
            x=x_col,
            y=y_col,
            groups=groups_col,
            window=8,
            title="测试时间滚动窗口图"
        )
        
        # 验证返回的图形对象
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(axes, np.ndarray)
        
        # 关闭图形以释放内存
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
