import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import random
from scipy.integrate import trapezoid  # 梯形积分
from scipy import stats
import os
root_path = os.path.abspath(os.path.dirname(__file__))

def evaluate_model_plot(y_true, y_predict, show=False):
    """
    y_true:
    y_predict:
    return  metrics
    """
    if show == True:
        # 图形基础设置
        plt.figure(figsize=(7, 5), dpi=400)
        plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        plt.grid(linestyle="--")  # 设置背景网格线为虚线
        ax = plt.gca()  # 获取坐标轴对象
        plt.scatter(y_true, y_predict, color='red')
        plt.plot(y_predict, y_predict, color='blue')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.xlabel("Measured", fontsize=12, fontweight='bold')
        plt.ylabel("Predicted", fontsize=12, fontweight='bold')
        plt.show()
        plt.figure(figsize=(7, 5), dpi=400)
        plt.hist(np.array(y_true) - np.array(y_predict), 40)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Residual", fontsize=20)
        plt.ylabel("Freq", fontsize=20)
        plt.show()
    from sklearn.metrics import mean_absolute_error
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    n = len(y_true)
    MSE = mean_squared_error(y_true, y_predict)
    RMSE = pow(MSE, 0.5)
    MAE = mean_absolute_error(y_true, y_predict)
    R2 = r2_score(y_true, y_predict)

    pccs = pearsonr(y_true, y_predict)[0]
    mre = mean_relative_error(y_true, y_predict)
    print("样本个数 ", round(n))
    print("均方根误差RMSE ", round(RMSE, 3))
    print("均方差MSE ", round(MSE, 3))
    print("平均绝对误差MAE ", round(MAE, 3))
    print("R2：", round(R2, 3))
    print("R：", round(pccs, 3))
    print("平均相对误差MRE：", round(mre, 3))
    return dict({"n": n, "MSE": MSE, "RMSE": RMSE, "MSE": MSE, "MAE": MAE, "R2": R2, 'R': pccs, 'MRE': mre})


def draw_feature_importance(feature, feature_importance):
    """
    features: name
    feature_importance:
    """
    # make importance relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(dpi=400)
    plt.barh(pos, list(feature_importance[sorted_idx]), align='center')
    plt.yticks(pos, list(feature[sorted_idx]), fontsize=6)
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.show()


def mean_relative_error(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: MRE
    """
    relative_error = np.average(np.abs(y_true - y_pred) / np.abs(y_true), axis=0)
    return relative_error


def top_train_test_split(df, y_col, test_ratio=0.2):
    n_train = round((1 - test_ratio) * df.shape[0])
    df_sorted = df.sort_values(by=[y_col])
    df_sorted = df_sorted.reset_index(drop=True)
    df_train = df_sorted[:n_train]
    df_test = df_sorted[n_train:]
    return df_train, df_test


def evaluation_top_val(y_train_predict, y_predict):
    # 预测值从[y_predict_min,max(y_train_predict,y_predict)] 分割 计算precision
    y_predict_min = min(y_predict)
    precision_value_list = []
    predict_value_list = []
    for predict_value in np.linspace(y_predict_min, max(max(y_train_predict), max(y_predict)), 30):
        precision_value = np.sum(y_predict >= predict_value) / (np.sum(y_predict >= predict_value) + np.sum(
            y_train_predict >= predict_value))
        precision_value_list.append(precision_value)  # y axis
        predict_value_list.append(predict_value)  # x axis

    scaled_predict_value = (predict_value_list - predict_value_list[0]) / (
            predict_value_list[-1] - (predict_value_list[0]))
    top_validation_score = trapezoid(precision_value_list, scaled_predict_value)
    return top_validation_score


def model_fit_evaluation(model, x_train, y_train, x_test, y_test, n_fold=5):
    """clf:
    x_train：训练集+验证集 用于计算交叉验证误差  np.array
    y_train： np.array
    x_test：计算测试误差
    n_fold：交叉验证折数 default = 5
    """
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
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


def generate_alloys_random(search_range, residual_element, category_col=[], samples=10, random_state=0):
    """
    search_range including lower bound and up bound of weight
    example:     search_range = {"Ag": [0.1, 1],
                    'Cu': [0.5, 1],
                    "Ni": [0.5, 1],
                    'In': [0.5, 1],
                    "Bi": [0.5, 1],
                    'condition1': [0, 1, 2, 3]
                    }
    residual_element: the main element to make sum weight% = 100
    :param samples:
    :param category_col:
    :param residual_element:
    :param search_range:
            samples: the number
    :return: dataframe
    """
    if not random_state is None:
        np.random.seed(random_state)
    search_range.pop(residual_element, None)
    rows = {}
    elements_col = [col for col in search_range.keys() if
                    len(search_range[col]) == 2 and len(col) <= 2 and not col in category_col]

    df_result = pd.DataFrame()
    for col in search_range.keys():
        if col in elements_col:  # elements features
            rows[col] = np.round(np.random.uniform(search_range[col][0], search_range[col][1], samples), 2)
        if col in category_col:  # category features
            rows[col] = np.random.choice(search_range[col], size=samples, replace=True)
    #
    # for i in range(samples):
    #
    #         if col in elements_col:  # elements features
    #             rows[col] = [round(random.uniform(search_range[col][0], search_range[col][1]), 2)]
    #         if col in category_col:  # category features
    #             rows[col] = random.sample(search_range[col], 1)

    result = pd.DataFrame(rows, columns=search_range.keys())
    result[residual_element] = 100
    for i in elements_col:
        result[residual_element] = round(result[residual_element] - result[i], 2)
    df_result = pd.concat([df_result, result])
    # delete invalid alloys
    df_result = df_result[df_result[residual_element] > 0]
    return df_result


def get_chemical_formula(dataset):
    """
    Al   Ni   Si
    0.5  0.5  0
    :return: get_chemical_formula from element mol weigh dataframe Al0.5Ni0.5
    """
    elements_columns = dataset.columns
    dataset = dataset.reset_index()
    chemistry_formula = []
    for i in range(dataset.shape[0]):
        single_formula = []
        for col in elements_columns:
            if (dataset.at[i, col]) > 0:
                # element
                single_formula.append(col)
                # ratio
                single_formula.append(str(dataset.at[i, col]))
        chemistry_formula.append("".join(single_formula))
    return chemistry_formula


def get_chemical_formula_from_weight(dataset):
    """
    H   O   C
    1  16  12
    :return: get_chemical_formula from element mol dataframe H1O1C1
    """
    weight_dict = get_weight_dict()
    elements_columns = dataset.columns
    dataset = dataset.reset_index()
    chemistry_formula = []
    for i in range(dataset.shape[0]):
        single_formula = []
        for element in elements_columns:
            if (dataset.at[i, element]) > 0:
                # element
                single_formula.append(element)
                # ratio
                single_formula.append(str(dataset.at[i, element] / weight_dict[element]))
        chemistry_formula.append("".join(single_formula))
    return chemistry_formula


def evaluation_top_val_by_percentile(y_train_predict, y_predict, percentiles=np.linspace(0, 100, 11), show=False):
    y_train_predict = np.array(y_train_predict)
    y_predict = np.array(y_predict)
    y_predict_min = min(y_predict)
    precision_value_list = []  # 大于某个阈值 y_predict的占比
    predict_value_list = []
    # y_train_predict >= y_predict_min
    train_higher_min = y_train_predict[np.where(y_train_predict >= y_predict_min)]
    array = np.array(list(train_higher_min) + list(y_predict))
    # print(array)
    for percentile in percentiles:
        predict_value = stats.scoreatpercentile(array, percentile)
        precision_value = np.sum(y_predict >= predict_value) / (np.sum(y_predict >= predict_value) + np.sum(
            y_train_predict >= predict_value))
        precision_value_list.append(precision_value)  # y axis
        predict_value_list.append(percentile / 100)  # x axis
    # print(predict_value_list)
    # print(precision_value_list)
    top_validation_score = trapezoid(precision_value_list, predict_value_list)
    if show:
        plt.plot(predict_value_list, precision_value_list)
        plt.xlabel("scaled(x)", fontsize=12, fontweight='bold')
        plt.ylabel("score(x)", fontsize=12, fontweight='bold')

        return top_validation_score, (predict_value_list, precision_value_list)
    return top_validation_score


def stratifed_sample(df, Y_columns, groups, train_split_ratio, random_state=None):
    """df:数据集
    Y_columns：分层name
    groups：拆分组数
    train_split_ratio：训练集合百分比
    """
    df_train = pd.DataFrame()
    ratio = train_split_ratio
    groups_list = [f'G{i}' for i in range(1, groups + 1)]
    df['groups'] = pd.qcut(df[Y_columns], groups, labels=groups_list, retbins=False, duplicates='raise')
    for i in groups_list:
        df_group = df[df.groups == i]  # 其中一个Grouop
        count = len(df_group)
        train_number = round(count * ratio)
        df_train = df_train.append(df_group.sample(train_number, random_state=random_state))
    # test index
    df_test_index = list(set(df.index) - set(df_train.index))
    # test_df
    df_test = df[df.index.isin(df_test_index)]
    return df_train, df_test


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def get_weight_dict():
    """ :return {'H': 1.00794, 'He': 4.002602, 'Li': 6.941}"""
    col = "MagpieData mean AtomicWeight"
    df = pd.read_excel(root_path + "/project_data/elements.xlsx", index_col="formula")
    return df[col].to_dict()


if __name__ == '__main__':
    """ test """
    pass
