"""
subgroup discovery by category
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_regression

# 生成模拟数据（10000样本，包含分类和数值特征）
np.random.seed(42)
data = {
    'region': np.random.choice(['North', 'South', 'East', 'West'], 10000),  # 分类变量
    'product_type': np.random.choice(['Electronics', 'Clothing', 'Books'], 10000),
    'user_age': np.random.randint(18, 65, 10000),  # 数值变量
    'browsing_time': np.random.normal(30, 10, 10000).astype(int),
    'purchase_amount': np.abs(np.random.normal(500, 200, 10000))  # 因变量（需预测的数值）
}
df = pd.DataFrame(data)

# 独热编码分类变量
encoder = OneHotEncoder()
cat_features = encoder.fit_transform(df[['region', 'product_type']])
#
numeric_features = ['user_age', 'browsing_time']
df_cat = pd.DataFrame(cat_features.toarray(), columns=encoder.get_feature_names_out())
df_encoded = pd.concat([df[numeric_features], df_cat], axis=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = df_encoded
y = df['purchase_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 拟合全局模型
rf_global = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_global.fit(X_train, y_train)
print(f"全局模型R²：{rf_global.score(X_test, y_test):.3f}")  # 示例输出：0.782

import matplotlib.pyplot as plt

# 获取特征重要性（参考网页7、8的基尼指数方法）
importance = rf_global.feature_importances_
features = X_train.columns
plt.barh(features, importance)
plt.title("feature importance")
plt.show()

# 筛选重要性前2的分类变量（示例假设为region_North和product_type_Electronics）
selected_cat = ['region_North', 'product_type_Electronics']

# 定义高潜力子群规则（例如：北方地区 & 电子产品）
subgroup_mask = (df['region'] == 'North') & (df['product_type'] == 'Electronics')
subgroup_data = df[subgroup_mask]
from itertools import product

# 自动遍历重要分类变量组合
regions = df['region'].unique()
products = df['product_type'].unique()

for region, product in product(regions, products):
    mask = (df['region'] == region) & (df['product_type'] == product)
    if sum(mask) < 500:  # 过滤样本量过小的子群
        continue
    subgroup_data = df[mask]
    # 拆分训练集
    X_sub = subgroup_data[numeric_features]  # 仅使用数值特征
    y_sub = subgroup_data['purchase_amount']
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_sub, y_sub, test_size=0.3)

    # 调优参数
    rf_subgroup = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=20)
    rf_subgroup.fit(X_train_sub, y_train_sub)
    print(f"{region} {product} R²: {rf_subgroup.score(X_test_sub, y_test_sub):.3f} ")
