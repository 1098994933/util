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

