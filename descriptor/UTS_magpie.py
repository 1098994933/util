"""
use magpie feature
"""
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import re
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)


def Linear_SVR(C=1.0, gamma=0.1, epsilon=1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVR(kernel="linear", C=C, gamma=gamma, epsilon=epsilon))
    ])


def RBF_SVR(C=1.0, gamma=1, epsilon=1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon))
    ])


def Poly_LinearRegression(degree=2):
    return Pipeline([('poly', PolynomialFeatures(degree=degree)),
                     ('linear', LinearRegression())])


def model_fit_evaluation(model, x_train, y_train, x_test, y_test, n_fold=5):
    """clf:
    x_train：训练集+验证集 用于计算交叉验证误差  np.array
    y_train： np.array
    x_test：计算测试误差
    n_fold：交叉验证折数 default = 5
    """
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=0)
    print(model)
    result = pd.DataFrame()
    for i, (train_index, test_index) in enumerate(kf.split(range(len(x_train)))):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_validation = x_train[test_index]  # get validation set
        y_validation = y_train[test_index]
        model.fit(x_tr, y_tr)

        result_subset = pd.DataFrame()  # save the prediction
        result_subset["y_validation"] = y_validation
        result_subset["y_pred"] = model.predict(x_validation)
        result = result.append(result_subset)
    print("cross_validation_error in validation set：")
    c = evaluate_model_plot(result["y_validation"], result["y_pred"], show=False)

    print("error in testing set：")
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    error_metric_testing = evaluate_model_plot(y_test, y_test_pred, show=False)  # 不画图
    print("====================================")
    return error_metric_testing

# def model_evaluation_by_custom_split(model, x_train, y_train, x_test, y_test, split_fun,**param):
#     """
#     传入一个自定义个分割方法 对x_train做切割 成x_train 和x_val
#     :param model:
#     :param x_train:
#     :param y_train:
#     :param x_test:
#     :param y_test:
#     :param split_fun:
#     :return:
#     """
#     print(model)
#     result = pd.DataFrame()
#     x_tr = split_fun(df_train)
#     y_tr = y_train[train_index]
#     x_validation = x_train[test_index]  # get validation set
#     y_validation = y_train[test_index]
#     model.fit(x_tr, y_tr)
#
#     result_subset = pd.DataFrame()  # save the prediction
#     result_subset["y_validation"] = y_validation
#     result_subset["y_pred"] = model.predict(x_validation)
#     result = result.append(result_subset)
#     print("cross_validation_error in validation set：")
#     c = evaluate_model_plot(result["y_validation"], result["y_pred"], show=False)
#
#     print("error in testing set：")
#     model.fit(x_train, y_train)
#     y_test_pred = model.predict(x_test)
#     error_metric_testing = evaluate_model_plot(y_test, y_test_pred, show=False)  # 不画图
#     return error_metric_testing

if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)
    elements_columns = config['elements_columns']
    dataset = pd.read_csv('dataset.csv')

    df_magpie = pd.read_csv('magpie_features.csv')
    magpie_features = list(df_magpie.columns[4:])
    print("magpie_features", magpie_features)
    dataset = pd.concat([dataset, df_magpie], axis=1)

    Y_col = 'Tensile Strength: Ultimate (UTS)'
    print(Y_col)
    features = elements_columns + ['condition2', 'condition3'] + magpie_features
    print(features)
    # ML
    ml_dataset = dataset[features + [Y_col]].dropna()

    X = ml_dataset[features]
    Y = ml_dataset[Y_col]
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    df_train, df_test = top_train_test_split(ml_dataset, Y_col, 0.03)
    X_train = df_train[features]
    X_test = df_test[features]
    Y_train = df_train[Y_col]
    Y_test = df_test[Y_col]

    linear_svr = Linear_SVR()
    rbf_svr = RBF_SVR()
    rigde = Ridge()
    lasso = Lasso()
    enr = ElasticNet()
    dtr = DecisionTreeRegressor()
    rfr = RandomForestRegressor()
    etr = ExtraTreesRegressor()
    abr = AdaBoostRegressor()
    gbr = GradientBoostingRegressor()
    sgd = SGDRegressor()
    lrg = LinearRegression()
    plr = Poly_LinearRegression()
    knr = KNeighborsRegressor()
    model_dict = {"linear_svr": linear_svr
        , "rbf_svr": rbf_svr
        , "LinearRegression": lrg
                  # ,"Poly_LinearRegression":plr
                  # ,"SGDRegressor":sgd
        , "Rigde": rigde
        , "Lasso": lasso
        , "ElasticNet": enr
        , "DecisionTree": dtr
        , "RandomForest": rfr
        , "ExtraTrees": etr
        , "AdaBoost": abr
        , "GradientBoosting": gbr
        , "KNeighborsRegressor": knr
                  }
    # record the error
    RMSE_list = []
    MAE_list = []
    R2_list = []
    R_list = []
    for model in model_dict.values():
        error_metric_testing = model_fit_evaluation(model, X_train.values, Y_train.values, X_test.values, Y_test.values,
                                                    n_fold=5)
        RMSE_list.append(error_metric_testing["RMSE"])
        MAE_list.append(error_metric_testing["MAE"])
        R2_list.append(error_metric_testing["R2"])
        R_list.append(error_metric_testing["R"])
    result_df = pd.DataFrame({"model": list(model_dict.keys()), "RMSE": RMSE_list,
                              "MAE": MAE_list, "R2": R2_list, 'R': R_list})
    print(result_df)

    print('Y_train max=', Y_train.max())
    print('Y_train min=', Y_test.min())

    model_final = GradientBoostingRegressor()
    model_final.fit(X_train, Y_train)
    y_predict = model_final.predict(X_test)
    y_true = Y_test

    # max里 有多少比训练集大
    print('查全率', np.sum(y_predict > Y_train.max()))
    print(y_predict.max())

    y_predict = y_predict
    evaluation_matrix = evaluate_model_plot(y_true, y_predict, show=False)
    plt.figure(figsize=(7, 5), dpi=400)
    plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()  # 获取坐标轴对象
    plt.scatter(y_true, y_predict, color='red')
    plt.plot([0, 700], [0, 700], color='blue')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel("Measured(UTS)", fontsize=12, fontweight='bold')
    plt.ylabel("Predicted(UTS)", fontsize=12, fontweight='bold')
    plt.xlim(0, 700)
    plt.ylim(0, 700)
    plt.text(0.1, 737.3, "$R^2={r2}$\n$MAE={MAE}$\n".format(r2=round(evaluation_matrix["R2"], 3)
                                                            , MAE=round(evaluation_matrix["MAE"], 3)
                                                            ))
    # plt.show()

