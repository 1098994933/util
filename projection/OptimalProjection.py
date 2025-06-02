import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import unittest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class OptimalProjection(object):
    """
    OptimalProjection: get the best projection from multiple projections
    最佳优化区识别算法：回归的话就是y大于平均为优类样本，否则需要指定优类样本的值,使用网格找到最优矩形区域
    算法复杂度O(P), P为投影个数
    """

    def __init__(self, projections, y, optimal_label=None, optimal_ratio_exponent=3, min_optimal_ratio=None):
        """
        Initialize the class with projections and target values.

        :param projections: List of PCA projection results, each element is a 2D array.
        :param y: Target values corresponding to each point in the projections.
        :param optimal_label: 优类样本的标签，用于分类数据，if it is a list of scale [target_min,target_max]
        :param optimal_ratio_exponent: control the area
        :param min_optimal_ratio: 区域内优类样本最小占比
        """
        # 数据检测机制
        if not isinstance(projections, list):
            raise ValueError("projections 必须是一个列表。")
        for projection in projections:
            if not isinstance(projection, np.ndarray) or projection.ndim != 2:
                raise ValueError("projections 列表中的每个元素必须是二维的 NumPy 数组。")
        if not isinstance(y, np.ndarray):
            raise ValueError("y 必须是一个 NumPy 数组。")
        if len(y) != projections[0].shape[0]:
            raise ValueError("y 的长度必须与 projections 中每个投影的样本数量一致。")
        if optimal_label is not None:
            if isinstance(optimal_label, list) and len(optimal_label) == 2:
                pass
            elif optimal_label not in y:
                raise ValueError("指定的 optimal_label 不在 y 中。")

        self.optimal_type = None  # 优类数据的类型 属于reg 或 scale 或 cls
        self.projections = projections
        self.y = y
        self.optimal_label = optimal_label
        if optimal_label is None:  # 判定为回归数据，则大于平均值为优类
            self.mean_y = np.mean(y)
            self.labels = np.array(y > self.mean_y).astype(int)
            self.optimal_type = "reg"
        elif isinstance(optimal_label, list) and len(optimal_label) == 2:  # 某个范围为优类
            target_min, target_max = optimal_label
            self.labels = np.array((y >= target_min) & (y <= target_max)).astype(int)
            self.optimal_type = "scale"
        else:
            self.labels = np.array(y == optimal_label).astype(int)
            self.optimal_type = "cls"
        if min_optimal_ratio is None:  # 则为优类样本比例 和 1 的平均值
            self.min_optimal_ratio = (sum(self.labels) / len(self.labels) + 1) / 2

        else:
            self.min_optimal_ratio = min_optimal_ratio
        self.best_x = None
        self.best_model = None
        self.best_rectangle = None  # 最佳投影区域
        self.optimal_ratio_exponent = optimal_ratio_exponent  # 越高则优化区域面积越小
        self.result = {}

    def find_best_projection(self):
        """
        Find the best projection that maximizes the score of rectangle area and the number of class 1 samples inside.
        :return: Index of the best projection and its corresponding score value.
        """
        best_index = 0
        best_product = 0

        num_grids = 20  # 可以调整网格数量

        for i, projection in enumerate(self.projections):
            x_min, x_max = np.min(projection[:, 0]), np.max(projection[:, 0])
            y_min, y_max = np.min(projection[:, 1]), np.max(projection[:, 1])
            total_area = (x_max - x_min) * (y_max - y_min)
            x_grid_size = (x_max - x_min) / num_grids
            y_grid_size = (y_max - y_min) / num_grids

            # 初始化网格计数
            grid_counts = np.zeros((num_grids, num_grids))
            for j in range(projection.shape[0]):
                if self.labels[j] == 1:
                    x_grid = int((projection[j, 0] - x_min) // x_grid_size)
                    y_grid = int((projection[j, 1] - y_min) // y_grid_size)
                    if 0 <= x_grid < num_grids and 0 <= y_grid < num_grids:
                        grid_counts[x_grid, y_grid] += 1

            max_score = 0
            best_rect = None
            for x1 in range(num_grids):
                for y1 in range(num_grids):
                    for x2 in range(x1 + 1, num_grids + 1):
                        for y2 in range(y1 + 1, num_grids + 1):
                            # 计算矩形的边界
                            rect_x_min = x_min + x1 * x_grid_size
                            rect_x_max = x_min + x2 * x_grid_size
                            rect_y_min = y_min + y1 * y_grid_size
                            rect_y_max = y_min + y2 * y_grid_size

                            # 计算矩形内的优类样本数
                            num_class_1 = np.sum(grid_counts[x1:x2, y1:y2])
                            support = num_class_1
                            # 计算矩形内的总样本数
                            in_rectangle = ((projection[:, 0] >= rect_x_min) & (projection[:, 0] <= rect_x_max) &
                                            (projection[:, 1] >= rect_y_min) & (projection[:, 1] <= rect_y_max)).sum()

                            if in_rectangle == 0:
                                score = 0
                                optimal_ratio = 0
                            else:
                                # 计算矩形区域面积
                                area = (rect_x_max - rect_x_min) * (rect_y_max - rect_y_min)
                                # score 为 计算矩形与总区域的面积比 * 优类样本率 ** （optimal_ratio_exponent）
                                optimal_ratio = num_class_1 / in_rectangle
                                score = optimal_ratio ** self.optimal_ratio_exponent * area / total_area
                            # 如果矩形面积大于0且优类样本率大于最小优类样本率，则更新最大得分和最佳索引
                            # 最小优类样本数（support）必须 > 总面积
                            min_support = sum(self.labels) * (x2 - x1) * (y2 - y1) / num_grids / num_grids
                            if score > max_score and optimal_ratio > self.min_optimal_ratio and support > min_support:
                                max_score = score
                                best_rect = (rect_x_min, rect_x_max, rect_y_min, rect_y_max)

            if max_score > best_product:
                best_product = max_score
                best_index = i
                self.best_x = projection
                self.best_rectangle = best_rect

        if self.best_x is not None and self.best_rectangle is not None:
            x_min, x_max, y_min, y_max = self.best_rectangle
            in_rectangle = ((self.best_x[:, 0] >= x_min) & (self.best_x[:, 0] <= x_max) &
                            (self.best_x[:, 1] >= y_min) & (self.best_x[:, 1] <= y_max)).sum()
            num_class_1 = ((self.best_x[:, 0] >= x_min) & (self.best_x[:, 0] <= x_max) &
                           (self.best_x[:, 1] >= y_min) & (self.best_x[:, 1] <= y_max) &
                           (self.labels == 1)).sum()
            area = (x_max - x_min) * (y_max - y_min)
            ratio = num_class_1 / in_rectangle if in_rectangle > 0 else 0
            self.result = {
                "total_samples": in_rectangle,
                "optimal_samples": num_class_1,
                "optimal_ratio": ratio,
                "optimal_area": area,
                "score": best_product
            }

        return best_index, best_product

    def plot_decision_boundary(self):
        """
        Plot the best rectangle on the best projection.
        """
        if self.best_x is None or self.best_rectangle is None:
            print("Please run find_best_projection() first.")
            return

        x_min, x_max, y_min, y_max = self.best_rectangle

        # 使用 seaborn 绘制散点图
        sns.scatterplot(x=self.best_x[:, 0], y=self.best_x[:, 1], hue=self.labels, palette={0: 'blue', 1: 'orange'},
                        edgecolor='k')
        plt.title('Optimal Projection')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        # Plot the rectangle
        rectangle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rectangle)

        # 确定优类样本标签名称
        if self.optimal_type == "reg":
            optimal_label_name = f"y > {self.mean_y:.2f}"
        elif self.optimal_type == "scale":
            target_min, target_max = self.optimal_label
            optimal_label_name = f"{target_min:.2f} < y < {target_max:.2f}"
        else:
            optimal_label_name = f"y={self.optimal_label})"

        # 手动添加图例
        handles, labels = plt.gca().get_legend_handles_labels()
        legend_labels = ['Non-Optimal Samples', f'Optimal Samples ({optimal_label_name})', 'Optimal Region']
        legend_handles = handles + [rectangle]
        plt.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    def get_rectangle_equations(self, pca, feature_names, scaler: StandardScaler = None):
        """
        Get the equations of the rectangle in terms of the original features.

        :param pca: Fitted PCA object from sklearn.
        :param feature_names: List of feature names corresponding to the original features.
        :param scaler: StandardScaler used for transform before pca
        :return: Equations of the rectangle in terms of the original features.
        """
        if self.best_rectangle is None:
            print("Please run find_best_projection() first.")
            return

        x_min, x_max, y_min, y_max = self.best_rectangle
        components = pca.components_[:2]
        equations = []

        if scaler is None:
            # 无标准化，直接使用PCA的components（假设数据已去均值）
            for i, (comp, name_prefix) in enumerate(zip(components, ["x", "y"])):
                terms = [f"{coef:.2f}*{feature_names[j]}" for j, coef in enumerate(comp)]
                eq = f"{x_min if i == 0 else y_min:.2f} ≤ {' + '.join(terms)} ≤ {x_max if i == 0 else y_max:.2f}"
                equations.append(eq)
            return equations
        else:
            # 有标准化，逆向转换回原始特征
            mean = scaler.mean_
            scale = scaler.scale_  # 标准差

            # 处理主成分1（x方向）
            comp_x = components[0]
            # 计算系数：components[i] / scale[j]
            coefs_x = comp_x / scale
            # 计算常数项：sum(components[i] * mean[j] / scale[j])
            const_x = np.dot(comp_x, mean / scale)
            lower_x = x_min + const_x
            upper_x = x_max + const_x
            terms_x = [f"{coef:.2f}*{feature_names[j]}" for j, coef in enumerate(coefs_x)]
            eq_x = f"{lower_x:.2f} ≤ {' + '.join(terms_x)} ≤ {upper_x:.2f}"
            equations.append(eq_x)

            # 处理主成分2（y方向）
            comp_y = components[1]
            coefs_y = comp_y / scale
            const_y = np.dot(comp_y, mean / scale)
            lower_y = y_min + const_y
            upper_y = y_max + const_y
            terms_y = [f"{coef:.2f}*{feature_names[j]}" for j, coef in enumerate(coefs_y)]
            eq_y = f"{lower_y:.2f} ≤ {' + '.join(terms_y)} ≤ {upper_y:.2f}"
            equations.append(eq_y)
            return equations


class TestOptimalProjection(unittest.TestCase):
    def test_find_best_projection_regression(self):
        """
        测试回归数据的最优投影查找
        """
        sample_num = 100
        projections = [np.random.rand(sample_num, 2) for _ in range(100)]
        y = np.random.rand(sample_num)
        op = OptimalProjection(projections, y)
        best_index, best_product = op.find_best_projection()
        self.assertEqual(isinstance(best_index, int), True)
        self.assertEqual(isinstance(best_product, float), True)
        self.assertTrue(0 <= best_index < len(projections))
        self.assertTrue(0 <= best_product)
        print(op.result)
        op.plot_decision_boundary()

    def test_find_best_projection_regression_scale(self):
        """
        测试指定范围的最优投影查找
        """
        sample_num = 100
        projections = [np.random.rand(sample_num, 2) for _ in range(100)]
        y = 10 * np.random.rand(sample_num)
        op = OptimalProjection(projections, y, optimal_label=[5, 10])
        best_index, best_product = op.find_best_projection()
        self.assertEqual(isinstance(best_index, int), True)
        self.assertEqual(isinstance(best_product, float), True)
        self.assertTrue(0 <= best_index < len(projections))
        self.assertTrue(0 <= best_product)
        print(op.result)
        op.plot_decision_boundary()

    def test_find_best_projection_classification(self):
        """
        测试分类数据的最优投影查找
        """
        sample_num = 100
        projections = [np.random.rand(sample_num, 2) for _ in range(100)]
        y = np.random.choice([0, 1], sample_num)
        optimal_label = 1
        op = OptimalProjection(projections, y, optimal_label, min_optimal_ratio=0.98)
        best_index, best_product = op.find_best_projection()
        op.plot_decision_boundary()
        self.assertEqual(isinstance(best_index, int), True)
        # self.assertEqual(isinstance(best_product, float), True)
        self.assertTrue(0 <= best_index < len(projections))
        self.assertTrue(0 <= best_product)
        print(op.result)
        op.plot_decision_boundary()

    def test_find_best_projection_range(self):
        """
        测试 optimal_label 为范围的情况
        """
        sample_num = 100
        projections = [np.random.rand(sample_num, 2) for _ in range(100)]
        y = np.random.rand(sample_num)
        optimal_label = [0.2, 0.8]
        op = OptimalProjection(projections, y, optimal_label)
        best_index, best_product = op.find_best_projection()
        self.assertEqual(isinstance(best_index, int), True)
        self.assertEqual(isinstance(best_product, float), True)
        self.assertTrue(0 <= best_index < len(projections))
        self.assertTrue(0 <= best_product)
        print(op.result)
        op.plot_decision_boundary()

    def test_invalid_projections_type(self):
        """
        测试 projections 不是列表的情况
        """
        sample_num = 100
        projections = np.random.rand(sample_num, 2)
        y = np.random.rand(sample_num)
        with self.assertRaises(ValueError):
            OptimalProjection(projections, y)

    def test_invalid_projection_dimension(self):
        """
        测试 projections 列表中的元素不是二维数组的情况
        """
        sample_num = 100
        projections = [np.random.rand(sample_num)]
        y = np.random.rand(sample_num)
        with self.assertRaises(ValueError):
            OptimalProjection(projections, y)

    def test_invalid_y_type(self):
        """
        测试 y 不是 NumPy 数组的情况
        """
        sample_num = 100
        projections = [np.random.rand(sample_num, 2) for _ in range(100)]
        y = [1] * sample_num
        with self.assertRaises(ValueError):
            OptimalProjection(projections, y)

    def test_mismatched_length(self):
        """
        测试 y 的长度与 projections 中每个投影的样本数量不一致的情况
        """
        sample_num = 100
        projections = [np.random.rand(sample_num, 2) for _ in range(100)]
        y = np.random.rand(sample_num - 1)
        with self.assertRaises(ValueError):
            OptimalProjection(projections, y)

    def test_invalid_optimal_label(self):
        """
        测试指定的 optimal_label 不在 y 中的情况
        """
        sample_num = 100
        projections = [np.random.rand(sample_num, 2) for _ in range(100)]
        y = np.random.choice([0, 1], sample_num)
        optimal_label = 2
        with self.assertRaises(ValueError):
            OptimalProjection(projections, y, optimal_label)

    def test_get_rectangle_equations(self):
        """
        测试 get_rectangle_equations 方法
        """
        sample_num = 100
        pca = PCA(n_components=2)
        # 这里需要一个合适的数据集来拟合 PCA，假设 data 是原始数据集
        x = np.random.rand(sample_num, 5)
        y = np.random.rand(sample_num)
        x_transformed = pca.fit_transform(x)
        feature_names = [f'feature_{i}' for i in range(x.shape[1])]
        projections = [x_transformed]

        op = OptimalProjection(projections, y)
        op.find_best_projection()
        op.plot_decision_boundary()
        equations = op.get_rectangle_equations(pca, feature_names)
        for equation in equations:
            print(equation)
        self.assertEqual(isinstance(equations, list), True)

    def test_multiple_features_pca(self):
        """测试多特征（>2）的方程生成"""
        sample_num = 99
        X = 100 * np.random.rand(sample_num, 3)
        y = np.random.choice([0, 1], sample_num)
        scaler = StandardScaler().fit(X)
        pca = PCA(n_components=2).fit(scaler.transform(X))
        projections = [pca.transform(scaler.transform(X))]  # 投影坐标为两点

        op = OptimalProjection(projections, y=y, optimal_label=1)
        op.find_best_projection()
        print(op.result)
        op.plot_decision_boundary()
        equations = op.get_rectangle_equations(pca, ["f1", "f2", "f3"], scaler)
        print(equations)
        # 验证系数数量与特征数一致
        for eq in equations:
            self.assertEqual(eq.count('*'), 3)  # 每个方程包含3个特征


if __name__ == "__main__":
    unittest.main()
