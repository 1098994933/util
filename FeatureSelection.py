"""
service for feature selection
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from scipy.stats import pearsonr


def select_features_by_mutual_info(x, y, n_features_to_select=5):
    # Compute mutual information between each feature and the target variable
    mi_scores = mutual_info_regression(x, y)

    # Select features with highest mutual information scores
    feature_indices = np.argsort(mi_scores)[::-1][:n_features_to_select]

    # Get feature names from original DataFrame
    feature_names = x.columns

    # Return selected feature names and data
    return feature_names[feature_indices], x.iloc[:, feature_indices]


def select_features_by_pcc(x, y, pcc):
    """
    :param x: pd.DataFrame
    :param y:
    :param pcc: float
    :return:
    """
    # remove no variance features
    variance = x.var(axis=0)
    zero_variance_cols = variance[variance == 0].index.tolist()
    x = x.drop(zero_variance_cols, axis=1)
    # calculate abs value of corr between features
    corr = abs(x.corr())
    features_names = list(x.columns)
    feature_num = x.shape[1]
    # set self.pcc == 0
    for i in range(feature_num):
        corr[features_names[i]][features_names[i]] = 0

    while True:
        # get max pcc
        max_corr = corr.max().max()

        if max_corr <= pcc or x.shape[1] < 2:  # stop when only one feature
            break

        # find the features should be deleted
        col1, row1 = np.where(corr == max_corr)

        pcc1 = abs(pearsonr(x[features_names[row1[0]]], y)[0])
        pcc2 = abs(pearsonr(x[features_names[col1[0]]], y)[0])
        if pcc1 >= pcc2:
            # print(pcc1, pcc2)
            drop_feature = corr.index[col1].tolist()[0]
        else:
            # print(pcc1, pcc2)
            drop_feature = corr.index[row1].tolist()[0]

        x = x.drop(drop_feature, axis=1)
        corr = abs(x.corr())
        feature_num = x.shape[1]
        features_names = list(x.columns)
        for i in range(feature_num):
            corr[features_names[i]][features_names[i]] = 0

    selected_features = list(x.columns)
    return selected_features


def select_features_by_rfe(model, x, y, features=None, step=None, cv=10):
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import RFECV
    if step is None:
        step = round(0.05 * x.shape[1])
    if features is None:
        features = np.array([i for i in range(x.shape[1])])
    if isinstance(x, pd.DataFrame):
        features = np.array(x.columns)
    feat_selector = RFECV(estimator=model,  # 学习器
                          step=step,  # 移除特征个数
                          cv=cv,  # 交叉验证次数
                          scoring='r2',  # 学习器的评价标准
                          verbose=1,
                          n_jobs=1
                          ).fit(x, y)
    rfe_rank_list = feat_selector.ranking_
    selected_features = np.array([features[i] for i in range(len(rfe_rank_list)) if (rfe_rank_list[i] == 1)])
    feat_selector.selected_features = selected_features
    return feat_selector


def variance_threshold_selector(data, threshold=0):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    support = selector.get_support()
    dropped_features = data.columns[~np.array(support)]  # ~是反运算符, 获得被筛掉的特征;显式地将布尔数组转换为 numpy 数组来避免类型检查工具的误报
    return data[data.columns[selector.get_support(indices=True)]], len(dropped_features)
