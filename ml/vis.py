"""
data visualization
"""
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from scipy.stats import pearsonr


def mean_relative_error(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: MRE
    """
    relative_error = np.average(np.abs(y_true - y_pred) / np.abs(y_true), axis=0)
    return relative_error


def cal_metric(y_true, y_predict):
    """
    y_true:
    y_predict:
    return  metrics MSE RMSE MAE R2 R MRE
    """
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    n = len(y_true)
    MSE = mean_squared_error(y_true, y_predict)
    RMSE = pow(MSE, 0.5)
    MAE = mean_absolute_error(y_true, y_predict)
    R2 = r2_score(y_true, y_predict)
    pccs = pearsonr(y_true, y_predict)[0]
    mre = mean_relative_error(y_true, y_predict)
    return dict({"n": n, "MSE": MSE, "RMSE": RMSE, "MSE": MSE, "MAE": MAE, "R2": R2, 'R': pccs, 'MRE': mre})


def true_predict_plot(y_true, y_predict, y_train, y_train_predict, show_metric=True):
    """
    vis for regression task
    :param show_metric:
    :param y_true: y of test data
    :param y_predict: y_predict of test data
    :param y_train: y_train
    :param y_train_predict:
    :return:
    """
    evaluation_metric = cal_metric(y_true, y_predict)

    lim_max = max(max(y_predict), max(y_true), max(y_train), max(y_train_predict)) * 1.02
    lim_min = min(min(y_predict), min(y_true), min(y_train), min(y_train_predict)) * 0.98
    plt.figure(figsize=(7, 5), dpi=400)
    plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()  # 获取坐标轴对象
    plt.scatter(y_true, y_predict, color='red', alpha=0.4, label='test')
    plt.scatter(y_train, y_train_predict, color='blue', alpha=0.4, label='train')
    plt.plot([lim_min, lim_max], [lim_min, lim_max], color='blue')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel("Measured", fontsize=12, fontweight='bold')
    plt.ylabel("Predicted", fontsize=12, fontweight='bold')
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    if show_metric:
        r2 = round(evaluation_metric["R2"], 3)
        MAE = round(evaluation_metric["MAE"], 3)
        r = round(evaluation_metric["R"], 3)
        ax.text(0, 1.1, f"$R^2={r2}$\n$MAE={MAE}$\n$R={r}$", transform=ax.transAxes)  # 使用相对坐标
    plt.legend()
    plt.show()
