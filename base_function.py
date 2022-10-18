import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import random
from scipy.integrate import trapezoid  # 梯形积分


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


def generate_alloys_random(search_range, residual_element, category_col=[], samples=10):
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
    search_range.pop(residual_element, None)
    rows = {}
    elements_col = [col for col in search_range.keys() if len(search_range[col]) == 2 and len(col)<=2 and not col in category_col]

    df_result = pd.DataFrame()
    for i in range(samples):
        for col in search_range.keys():
            if col in elements_col:  # elements features
                rows[col] = [round(random.uniform(search_range[col][0], search_range[col][1]), 2)]
            if col in category_col:  # category features
                rows[col] = random.sample(search_range[col], 1)

        result = pd.DataFrame(rows, columns=search_range.keys())
        result[residual_element] = 100
        for i in elements_col:
            result[residual_element] = round(result[residual_element] - result[i], 2)
        df_result = pd.concat([df_result, result])
    return df_result


def get_chemical_formula(dataset):
    """
    Al   Ni   Si
    0.5  0.5  0
    :return: get_chemical_formula from element weigh dataframe Al0.5Ni0.5
    """
    elements_columns = dataset.columns
    dataset = dataset.reset_index()
    chemistry_formula = []
    atoms_weight = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1791, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386, 'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078, 'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938045, 'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.64, 'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585, 'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.96, 'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116, 'Pr': 140.90765, 'Nd': 144.242, 'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25, 'Td': nan, 'Dy': 162.5, 'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.054, 'Lu': 174.9668, 'Hf': 178.49, 'Ta': 180.94788, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804, 'Po': 209.0, 'At': 210.0, 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.03806, 'Pa': 231.03586, 'U': 238.02891, 'Np': 237.0, 'Pu': 244.0, 'Am': 243.0, 'Cm': 247.0, 'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0, 'Md': 258.0, 'No': 259.0, 'Lr': 262.0, 'Rf': 265.0, 'Db': 268.0, 'Sg': 271.0}
    for i in range(dataset.shape[0]):
        single_formula = []
        sum_mol = 0  # sum mol of elements
        for col in elements_columns:
            if (dataset.at[i, col]) > 0:
                mol = dataset.at[i, col]/atoms_weight[col]
                sum_mol = sum_mol + mol
        for col in elements_columns:
            if (dataset.at[i, col]) > 0:
                mol = dataset.at[i, col] / atoms_weight[col]
                single_formula.append(col)
                single_formula.append(str(100 * mol/sum_mol))
        chemistry_formula.append("".join(single_formula))
    return chemistry_formula



