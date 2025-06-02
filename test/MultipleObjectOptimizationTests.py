import unittest
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from sklearn.ensemble import RandomForestRegressor


class TestMultipleObjectOptimization(unittest.TestCase):
    """多目标优化测试类
    
    测试多目标优化问题的定义、求解和结果验证。
    包括问题初始化、评估、优化过程和帕累托前沿的测试。
    """
    @classmethod
    def setUpClass(cls):
        """准备测试数据和模型"""
        np.random.seed(42)
        
        # 生成训练数据
        cls.X1 = np.random.uniform(0, 10, 100).reshape(-1, 1)  # 共享特征
        cls.X2a = np.random.uniform(0, 5, 100).reshape(-1, 1)  # 模型1独特征
        cls.X2b = np.random.uniform(-5, 0, 100).reshape(-1, 1)  # 模型2独特征
        
        # 生成目标值（确保是一维数组）
        cls.y1 = (2 * cls.X1 + 3 * cls.X2a).ravel()   # 线性模型
        cls.y2 = (0.5 * cls.X1 ** 2 + 4 * cls.X2b).ravel()  # 二次模型
        
        # 训练模型
        cls.model1 = RandomForestRegressor(random_state=42)
        cls.model1.fit(np.hstack([cls.X1, cls.X2a]), cls.y1)
        
        cls.model2 = RandomForestRegressor(random_state=42)
        cls.model2.fit(np.hstack([cls.X1, cls.X2b]), cls.y2)

    def setUp(self):
        """每个测试用例前的准备工作"""
        self.problem = MLModelProblem(
            X1=self.X1,
            X2a=self.X2a,
            X2b=self.X2b,
            model1=self.model1,
            model2=self.model2
        )
        self.algorithm = NSGA2(
            pop_size=100,
            eliminate_duplicates=True
        )

    def test_problem_initialization(self):
        """测试问题初始化
        
        验证优化问题的变量数量、目标数量、边界条件等是否正确设置。
        """
        self.assertEqual(self.problem.n_var, 3)  # 3个变量
        self.assertEqual(self.problem.n_obj, 2)  # 2个目标
        self.assertEqual(len(self.problem.xl), 3)  # 3个下界
        self.assertEqual(len(self.problem.xu), 3)  # 3个上界

    def test_problem_evaluation(self):
        """测试问题评估
        
        验证优化问题的评估函数是否正确计算目标值。
        """
        # 测试单个解
        x = np.array([[5.0, 2.5, -2.5]])  # 一个测试点
        out = {}
        self.problem._evaluate(x, out)
        
        # 验证输出形状
        self.assertEqual(out["F"].shape, (1, 2))
        
        # 验证目标值在合理范围内
        self.assertTrue(np.all(out["F"] <= 0))  # 因为我们要最大化，所以原始值应该是负数

    def test_optimization(self):
        """测试优化过程
        
        验证优化算法是否能够正确求解多目标优化问题。
        """
        res = minimize(
            self.problem,
            self.algorithm,
            ('n_gen', 50),  # 减少代数以加快测试
            seed=42,
            verbose=False
        )
        
        # 验证结果
        self.assertIsNotNone(res.X)  # 确保有解
        self.assertIsNotNone(res.F)  # 确保有目标值
        self.assertEqual(res.X.shape[1], 3)  # 3个变量
        self.assertEqual(res.F.shape[1], 2)  # 2个目标

    def test_pareto_front(self):
        """测试帕累托前沿
        
        验证优化结果是否形成有效的帕累托前沿。
        """
        res = minimize(
            self.problem,
            self.algorithm,
            ('n_gen', 50),
            seed=42,
            verbose=False
        )
        
        # 验证帕累托前沿的性质
        F = -res.F  # 转回原始目标值（最大化）
        
        # 检查是否有非支配解
        self.assertGreater(len(F), 0)
        
        # 检查目标值是否在合理范围内
        self.assertTrue(np.all(F[:, 0] >= self.y1.min()))
        self.assertTrue(np.all(F[:, 0] <= self.y1.max()))
        self.assertTrue(np.all(F[:, 1] >= self.y2.min()))
        self.assertTrue(np.all(F[:, 1] <= self.y2.max()))

    def test_visualization(self):
        """测试可视化功能
        
        验证帕累托前沿的可视化是否正确。
        """
        res = minimize(
            self.problem,
            self.algorithm,
            ('n_gen', 50),
            seed=42,
            verbose=False
        )
        
        # 创建图形
        plt.figure(figsize=(8, 6))
        plt.scatter(-res.F[:, 0], -res.F[:, 1], c='blue', alpha=0.7, label='Pareto Front')
        plt.xlabel('Model 1 Output', fontsize=12)
        plt.ylabel('Model 2 Output', fontsize=12)
        plt.title('Pareto Front of Multi-objective Optimization', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 验证图形是否成功创建
        self.assertIsNotNone(plt.gcf())
        plt.close()


class MLModelProblem(Problem):
    """多目标优化问题定义"""
    def __init__(self, X1, X2a, X2b, model1, model2):
        # 自变量范围
        x_scale_min = np.array([X1.min(), X2a.min(), X2b.min()])
        x_scale_max = np.array([X1.max(), X2a.max(), X2b.max()])
        
        super().__init__(
            n_var=3,  # 3个变量
            n_obj=2,  # 2个目标
            xl=x_scale_min,
            xu=x_scale_max,
            type_var=np.float64
        )
        
        self.model1 = model1
        self.model2 = model2

    def _evaluate(self, x, out, *args, **kwargs):
        # 分解变量
        X1_val = x[:, [0]]
        X2a_val = x[:, [1]]
        X2b_val = x[:, [2]]

        # 预测目标值
        y1_predict = self.model1.predict(np.hstack([X1_val, X2a_val]))
        y2_predict = self.model2.predict(np.hstack([X1_val, X2b_val]))

        # 转为负数（因为pymoo默认最小化，而我们要最大化目标）
        out["F"] = np.column_stack([-y1_predict, -y2_predict])


if __name__ == '__main__':
    unittest.main()
