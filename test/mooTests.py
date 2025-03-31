import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# 1. 模拟训练数据（假设原始数据范围）
np.random.seed(42)
X1 = np.random.uniform(0, 10, 100).reshape(-1, 1)  # 共享特征
X2a = np.random.uniform(0, 5, 100).reshape(-1, 1)  # 模型1独特征
X2b = np.random.uniform(-5, 0, 100).reshape(-1, 1)  # 模型2独特征
y1 = 2 * X1 + 3 * X2a   # 线性模型
y2 = 0.5 * X1 ** 2 + 4 * X2b  # 二次模型

model1 = RandomForestRegressor()
model1.fit(np.hstack([X1, X2a]), y1)

model2 = RandomForestRegressor()
model2.fit(np.hstack([X1, X2b]), y2)


# 2. 定义多目标优化问题
class MLModelProblem(Problem):
    def __init__(self):
        # 自变量范围
        x_scale_min = np.array([X1.min(), X2a.min(), X2b.min()])  # x1, x2a, x2b
        x_scale_max = np.array([X1.max(), X2a.max(), X2b.max()])
        super().__init__(n_var=3, n_obj=2, xl=x_scale_min, xu=x_scale_max, type_var=np.float64)

    def _evaluate(self, x, out, *args, **kwargs):
        # 分解变量：x1（共享）, x2a（模型1）, x2b（模型2）
        X1_val = x[:, [0]]
        X2a_val = x[:, [1]]
        X2b_val = x[:, [2]]

        y1_predict = model1.predict(np.hstack([X1_val, X2a_val]))
        f1 = y1_predict

        y2_predict = model2.predict(np.hstack([X1_val, X2b_val]))
        f2 = y2_predict

        # pymoo默认最小化，转为负数（因为要最大化目标）
        print(f1.shape, f2.shape)
        out["F"] = np.column_stack([-f1, -f2])



# 3. 配置优化算法
problem = MLModelProblem()
algorithm = NSGA2(
    pop_size=100,
    eliminate_duplicates=True
)

# 4. 执行优化
res = minimize(
    problem,
    algorithm,
    ('n_gen', 100),
    seed=42,
    verbose=True
)

# 5. 结果可视化
plt.figure(figsize=(8, 6))
plt.scatter(-res.F[:, 0], -res.F[:, 1], c='blue', alpha=0.7, label='parato')
plt.xlabel('y1', fontsize=12)
plt.ylabel('y2', fontsize=12)
plt.title('parato', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# 6. 最优解示例
print("最优解示例：")
for i in range(3):
    print(f"solution {i + 1}:")
    print(f"  变量: x1={res.X[i, 0]:.2f}, x2a={res.X[i, 1]:.2f}, x2b={res.X[i, 2]:.2f}")
    print(f"  目标: 模型1 R²={-res.F[i, 0]:.4f}, 模型2 R²={-res.F[i, 1]:.4f}\n")
