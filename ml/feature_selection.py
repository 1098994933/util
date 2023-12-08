"""
feature selection method
interface:
    X dataframe or numpy
    Y

"""


def fs_by_boruta(model, x, y, features=None, verbose=2, random_state=None):
    """
    # pip install Boruta
    feature selection by boruta
    x: np array
    y: np array
    self.selected_features to get features
    return
    """
    import numpy as np
    import pandas as pd
    from boruta import BorutaPy

    if features is None:
        features = np.array([i for i in range(x.shape[1])])
    if isinstance(x, pd.DataFrame):
        features = np.array(x.columns)
    # define Boruta feature selection method
    feat_selector = BorutaPy(model, n_estimators='auto', verbose=verbose, random_state=random_state)
    feat_selector.fit(x, y)
    # X_filtered = feat_selector.transform(X)
    selected_features = features[feat_selector.support_]
    feat_selector.selected_features = selected_features
    return feat_selector


def fs_by_rfe(model, x, y, features=None, step=None, cv=5, scoring='r2'):
    """
    feature selection by RFE
    :param model:
    :param x:
    :param y:
    :param features:
    :param step:
    :param cv:
    :param scoring:
    :return:
    """
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import RFECV
    if step:
        # at least 1
        step = max(round(0.05 * x.shape[1]), 1)
    if features is None:
        features = np.array([i for i in range(x.shape[1])])
    if isinstance(x, pd.DataFrame):
        features = np.array(x.columns)
    feat_selector = RFECV(estimator=model,  # 学习器
                          step=step,  # 移除特征个数
                          cv=cv,  # 交叉验证次数
                          scoring=scoring,  # 学习器的评价标准
                          verbose=1,
                          n_jobs=4
                          ).fit(x, y)
    rfe_rank_list = feat_selector.ranking_
    selected_features = np.array([features[i] for i in range(len(rfe_rank_list)) if (rfe_rank_list[i] == 1)])
    feat_selector.selected_features = selected_features
    return feat_selector
