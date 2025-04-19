# 特征筛选类
import logging
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif, mutual_info_regression, RFE, RFECV
from sklearn.metrics import mutual_info_score, r2_score
import pickle
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SequentialFeatureSelector


class FeatureSelector(object):
    """
    FeatureSelector: A class for performing feature selection on a dataset (x,y).
    """

    def __init__(self, x: pd.DataFrame, y, select_method, selector_params=None):
        """
        """
        self.select_method_name = None  # 选择方法名称(Chinese）
        self.mutual_info_threshold = None
        self.x = x
        self.y = y
        self.selector_params = selector_params  # 筛选超参数
        self.original_features = list(x.columns)  # 原始特征名称
        self.selected_features = None  # 选择的特征名称
        if selector_params is None:
            self.selector_params = {'pcc': 0.95, 'mutual_info_threshold': 0.1, "n_feature": 20}  # 特征选择参数
        self.selector = None
        self.select_method = select_method
        # 特征选择结果
        self.selector_result = {
        }
        # model-based feature selection properties.
        self.cv_score = None

    def fit(self, method='correlation'):
        """
        根据给定的数据 X 和标签 y 进行特征选择。
        """
        # 确保输入是 DataFrame
        X = self.x
        y = self.y
        method = self.select_method
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.original_features = X.columns.tolist()

        if method == 'pcc':
            self.select_features_by_pcc(X, y, selector_params=self.selector_params)
        elif method == "mutual_info":
            pass
            # self.mutual_info_threshold = self.selector_params['mutual_info_threshold']
            # self.select_features_by_mutual_info(X, y)
        elif method == "mrmr":
            self.select_features_by_mrmr()
        # model-based feature selection
        elif method == "sfs":
            self.select_features_by_SequentialFeatureSelector(X, y)

    def select_features_by_pcc(self, x, y, selector_params=None):
        """
        使用皮尔逊相关系数进行特征选择，并记录被筛选掉的特征与保留特征的相关性
        
        params:
        x: pd.DataFrame, 特征数据
        y: pd.DataFrame, pd.Series, np.ndarray, 目标变量
        selector_params: dict, 选择参数，包含pcc阈值
        
        return:
        list: 选择的特征列表
        """
        self.select_method = "相关性筛选"
        self.select_method_name = "相关性筛选"
        if selector_params is not None:
            self.selector_params = selector_params
        pcc = self.selector_params.get('pcc', 0.95)
        
        # 确保x是DataFrame
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
            
        # 确保y是Series
        if isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                raise ValueError("y should be a single column DataFrame")
            y = y.iloc[:, 0]
        elif isinstance(y, np.ndarray):
            if len(y.shape) > 1 and y.shape[1] > 1:
                raise ValueError("y should be a 1D array")
            y = pd.Series(y.ravel())
        
        # 初始化结果字典
        pcc_result = {}
        
        # 移除方差为0的特征
        variance = x.var(axis=0)
        zero_variance_cols = variance[variance == 0].index.tolist()
        x = x.drop(zero_variance_cols, axis=1)
        
        # 计算特征间的相关系数矩阵（只计算一次）
        corr_matrix = abs(x.corr())
        features_names = list(x.columns)
        feature_num = x.shape[1]
        
        # 初始化被删除的特征列表
        dropped_features = []
        
        # 设置对角线为0
        np.fill_diagonal(corr_matrix.values, 0)
        
        # 创建特征掩码，用于跟踪哪些特征被保留
        feature_mask = np.ones(feature_num, dtype=bool)
        
        while True:
            # 获取当前保留特征的相关系数矩阵
            current_corr = corr_matrix.iloc[feature_mask, feature_mask]
            
            # 获取最大相关系数
            max_corr = current_corr.max().max()
            
            if max_corr <= pcc or sum(feature_mask) < 2:
                break
                
            # 找到相关系数最大的特征对
            max_corr_idx = np.unravel_index(current_corr.values.argmax(), current_corr.shape)
            current_features = current_corr.index.tolist()
            feature1 = current_features[max_corr_idx[0]]
            feature2 = current_features[max_corr_idx[1]]
            
            # 计算与目标变量的相关性
            pcc1 = abs(pearsonr(x[feature1], y)[0])
            pcc2 = abs(pearsonr(x[feature2], y)[0])
            
            if pcc1 >= pcc2:
                keep_feature = feature1
                drop_feature = feature2
            else:
                keep_feature = feature2
                drop_feature = feature1
                
            # 记录被删除的特征与保留特征的关系，包括相关系数
            if keep_feature not in pcc_result:
                pcc_result[keep_feature] = []
            
            # 计算相关系数并保留3位有效数字
            correlation = float(round(x[keep_feature].corr(x[drop_feature]), 3))
            
            # 记录特征名和相关系数
            pcc_result[keep_feature].append({
                'feature': drop_feature,
                'correlation': correlation
            })
            
            dropped_features.append(drop_feature)
            
            # 更新特征掩码
            drop_idx = features_names.index(drop_feature)
            feature_mask[drop_idx] = False
            
        # 获取最终选择的特征
        selected_features = [f for f, m in zip(features_names, feature_mask) if m]
        self.selected_features = selected_features
        
        # 更新selector_result
        self.selector_result = {
            "selected_features": selected_features,
            "dropped_features": dropped_features,
            "pcc_result": pcc_result,
            "pcc_threshold": pcc
        }
        
        return selected_features

    def select_features_by_mutual_info(self, X, y):
        """
        使用互信息进行特征选择。
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 使用互信息计算每个特征的重要性
        mutual_info_scores = mutual_info_regression(X, y)

        # 获取特征名称
        feature_names = X.columns.tolist()

        # 将得分与特征名称配对
        feature_importance = list(zip(feature_names, mutual_info_scores))

        # 排序以获取最高得分的特征
        sorted_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)

        # 选择互信息大于阈值的特征
        selected_features = [feature for feature, score in sorted_features if score >= self.mutual_info_threshold]

        self.selected_features = selected_features
        return selected_features

    def select_features_by_mrmr(self):
        """
        使用最大相关最小冗余（mRMR）方法进行特征选择。
        """
        self.select_method = "最大相关最小冗余"
        self.select_method_name = "最大相关最小冗余"
        X = self.x
        y = self.y
        k = int(self.selector_params.get('n_feature', 20))
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        selected_features = []
        candidate_features = X.columns.tolist()

        while len(selected_features) < k and candidate_features:
            mi_scores = []
            for feature in candidate_features:
                # 计算与目标变量的互信息
                relevance = mutual_info_score(X[feature], y)
                redundancy = np.mean([mutual_info_score(X[feature], X[selected_feature]) for selected_feature in
                                      selected_features]) if selected_features else 0
                mi_score = relevance - redundancy
                mi_scores.append((feature, mi_score))

            # 选择最大mRMR得分的特征
            best_feature = max(mi_scores, key=lambda x: x[1])[0]
            selected_features.append(best_feature)
            candidate_features.remove(best_feature)

        self.selected_features = selected_features
        return selected_features

    def select_features_by_forward_selection(self, model, x, y, selector_params, initial_features=None):
        """  暂时未使用
        Select features using forward selection.

        Parameters:
        model (sklearn.base.BaseEstimator): The machine learning model to use for evaluation.
        x (pd.DataFrame or np.ndarray): The input data with features.
        y (pd.Series or np.ndarray): The target variable.
        initial_features (list[str] or list[int], optional): Initial set of features to start the selection process. If None, an empty set is used. Default: None.
        step (int, optional): Number of features to add in each iteration. Default: 1.
        cv (int, optional): Number of folds for cross-validation. Default: 10.

        Returns:
        list[str]: Selected features after forward selection.
        """
        cv = selector_params.get("cv", 5)

        if not isinstance(cv, int) or cv < 2:
            raise ValueError("Cross-validation fold number (cv) must be an integer greater than 1.")

        if model is None:
            model = GradientBoostingRegressor()

        model_name = model.__class__.__name__

        if initial_features is None:
            initial_features = []
        else:
            assert all([isinstance(feat, str) or isinstance(feat, int) for feat in initial_features])
        x = pd.DataFrame(x)
        remaining_features = [col for col in x.columns if col not in initial_features]

        best_score = 0
        selected_features = initial_features.copy()
        step_info = []  # 用于记录每个 step 的信息

        # 算法：增加特征，如果分数增加，则增加到选择特征，否则remove
        end_status = False
        while not end_status:
            scores = {}
            for new_feature in remaining_features.copy():
                current_features = selected_features + [new_feature]
                y_predict_cv = cross_val_predict(model, x[current_features], y, cv=cv, n_jobs=-1)
                score = r2_score(y, y_predict_cv)
                scores[new_feature] = score
                if best_score < score:
                    selected_features.append(new_feature)
                    best_score = score
                    remaining_features.remove(new_feature)
                    print("add", new_feature, best_score)
                else:
                    continue
            end_status = True

        self.select_method = "向前选择法"
        self.select_method_name = "向前选择法"
        self.selected_features = selected_features
        self.selector_result = {
            "selected_features": selected_features,
            "best_score_r2": best_score,
            "model_name": model_name,
            "step_info": step_info
        }
        return selected_features

    def transform(self, X):
        """
        应用特征选择结果到新的数据集 X 上。
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X[self.selected_features]

    def select_features_by_SequentialFeatureSelector(self, estimator, x, y, direction='forward', selector_params={},
                                                     n_features_to_select='auto', tol=None, scoring=None):
        """
        feature selection by SequentialFeatureSelector from sklearn
        """
        self.select_method = "向前选择法"
        self.select_method_name = "向前选择法"
        cv = selector_params.get("cv", 5)
        estimator.fit_cls_model(x, y)
        sfs = SequentialFeatureSelector(estimator, n_features_to_select=n_features_to_select, cv=cv,
                                        direction=direction)
        sfs.fit(x, y)
        selected_features = list(sfs.get_feature_names_out())
        self.selected_features = selected_features
        y_predict = cross_val_predict(estimator, x[selected_features], y, cv=cv)
        self.cv_score = round(r2_score(y, y_predict), 3)
        model_name = estimator.__class__.__name__
        self.selector_result = {
            "selected_features": selected_features,
            "best_score_r2": self.cv_score,
            "model_name": model_name,
        }
        return selected_features

    def select_features_by_auto(self):
        """
        auto feature selection
        1. select by pcc
        2. select by select_features_by_mutual_info if features >20
        3. sfs
        """

        # 1. select by pcc
        selected_features = self.select_features_by_pcc(self.x, self.y, selector_params=self.selector_params)
        # 2. select by mutual_info if features >20
        if len(selected_features) > 20:
            selected_features = self.select_features_by_mrmr()
        # 3. feature number should>=2
        if len(selected_features) >= 2:
            selected_features = self.select_features_by_SequentialFeatureSelector(GradientBoostingRegressor(),
                                                                                  self.x[selected_features], self.y)
        else:
            pass
        cv_fold = min(self.x.shape[0], 5)
        y_predict_cv = cross_val_predict(GradientBoostingRegressor(), self.x[selected_features], self.y, cv=cv_fold)
        score = round(r2_score(self.y, y_predict_cv), 3)

        self.selected_features = selected_features
        self.select_method = "智能化筛选"
        self.select_method_name = "智能化筛选"
        self.cv_score = round(score, 2)
        self.selector_result = {
            "selected_features": self.selected_features,
            "best_score_r2": self.cv_score,
            "method": self.select_method,
            "best_model": "GradientBoostingRegressor",
            "best_model_name": "梯度上升回归",
        }
        return selected_features

    def select_features_by_RFECV(self, estimator, x, y, selector_params={}, scoring="r2"):
        """
        feature selection by Recursive Feature Elimination with Cross - Validation from sklearn
        """
        self.select_method = "特征迭代消除法"
        self.select_method_name = "特征迭代消除法"
        cv = selector_params.get("cv", 5)
        step = selector_params.get("step", 1)
        estimator.fit_cls_model(x, y)

        fs = RFECV(estimator, cv=cv, step=step, scoring=scoring)
        fs.fit(x, y)
        selected_features = list(x.columns[fs.support_])
        self.selected_features = selected_features
        y_predict = cross_val_predict(estimator, x[selected_features], y, cv=cv)
        self.cv_score = round(r2_score(y, y_predict), 3)
        model_name = estimator.__class__.__name__
        self.selector_result = {
            "selected_features": selected_features,
            "best_score_r2": self.cv_score,
            "model_name": model_name,
            "optimal_num_features": len(selected_features)
        }
        return selected_features
    def save_result(self, filename):
        """
        save result data as a config file
        """
        logging.info("Saving feature selection result to %s", filename)
        with open(filename, 'wb') as file:
            pickle.dump({
                'original_features': self.original_features,
                'selected_features': self.selected_features,
                'selector_params': self.selector_params,
                "select_method": self.select_method,
                "select_result": self.selector_result,
                "select_method_name": self.select_method_name,
                "cv_score": self.cv_score
            }, file)

    @staticmethod
    def load_result(filename):
        """
        load feature selection result from pkl.
        """
        with open(filename, 'rb') as file:
            config = pickle.load(file)
        return config

    def pcc_result_to_dataframe(self) -> pd.DataFrame:
        """
        将pcc_result转换为DataFrame格式
        
        return:
        pd.DataFrame: 包含三列的DataFrame
            - kept_feature: 保留的特征
            - correlated_features: 相关的特征列表
            - correlations: 相关系数列表
        """
        if not hasattr(self, 'selector_result') or 'pcc_result' not in self.selector_result:
            raise ValueError("No pcc_result found. Please run select_features_by_pcc first.")
            
        pcc_result = self.selector_result['pcc_result']
        selected_features = self.selector_result['selected_features']
        
        # 初始化结果列表
        result_data = []
        
        # 处理每个保留特征
        for feature in selected_features:
            if feature in pcc_result:
                # 获取相关特征和相关系数
                correlated_features = [item['feature'] for item in pcc_result[feature]]
                correlations = [item['correlation'] for item in pcc_result[feature]]
            else:
                # 如果没有相关特征，使用空列表
                correlated_features = []
                correlations = []
                
            result_data.append({
                'kept_feature': feature,
                'correlated_features': correlated_features,
                'correlations': correlations
            })
        
        # 创建DataFrame
        df = pd.DataFrame(result_data)
        
        return df
