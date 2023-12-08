"""
eda analysis

conclusion:
1.
2.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import numpy as np
import pickle
import pickle

from scipy import stats
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import os
import scipy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
# from util.alg import Linear_SVR, RBF_SVR

val_config = {

}
if __name__ == '__main__':
    dataset = pd.read_excel("data/dataml1.xlsx")
    Y_col = 'Y'
    features = list(dataset.columns[2:])
    print(len(dataset[Y_col]))
    print(min(dataset[Y_col]), max(dataset[Y_col]))
    print(dataset[Y_col].std())
    with plt.style.context([]):
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        plt.hist(dataset[Y_col], bins=10)
        plt.savefig(f'./figures/hist_{Y_col}.png', bbox_inches='tight')
    print(dataset)

    ml_dataset = dataset
    # ml
    X = ml_dataset[features]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    Y = ml_dataset[Y_col]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)
    # from sklearn.feature_selection import VarianceThreshold
    # var = VarianceThreshold(threshold=0)
    # X = var.fit_transform(X)
    print(X_train.shape, Y_train.shape)
    X_train = pd.DataFrame(X_train, columns=features)
    columns_with_nulls = X_train.columns[X_train.isnull().any()]
    print(columns_with_nulls)
    feature_selection = SelectKBest(f_regression, k=200).fit(X_train, Y_train)

    feature_scores = feature_selection.scores_
    print('feature_scores:', feature_scores)  # 得分越高，特征越重要
    indices = np.argsort(feature_scores)[::-1]
    val_config['feature_num'] = len(features)
    best_features = list(X_train.columns.values[indices[0:val_config['feature_num']]])
    print("best_features", best_features)
    X_train = feature_selection.transform(X_train)
    X_test = feature_selection.transform(X_test)
    sc = MinMaxScaler()
    alg_dict = {
        # "Lasso": Lasso(),
        # "Ridge": Ridge(),
        # "LinearRegression": LinearRegression(),
        # 'LinearSVR': Linear_SVR(C=1),
        # 'LinearSVR2': Linear_SVR(C=100),
        # 'LinearSVR3': Linear_SVR(C=10),
        # "GradientBoosting": GradientBoostingRegressor(),
        # "AdaBoost": AdaBoostRegressor(),
        # "ExtraTrees": ExtraTreesRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=100, min_samples_split=2, random_state=4, n_jobs=4),
        # "KNeighbors": KNeighborsRegressor(),
        # "DecisionTree": DecisionTreeRegressor(),
        # 'RbfSVR': RBF_SVR(C=1),
        # 'RbfSVR1': RBF_SVR(C=10, gamma=0.20),
        # 'RbfSVR2': RBF_SVR(C=100, gamma=0.10),
        # 'RbfSVR3': RBF_SVR(C=1000, gamma=0.05),
        # 'RbfSVR4': RBF_SVR(C=0.1, gamma=0.01),
    }
    best_model = None
    best_score = 10 ** 10
    for alg_name in alg_dict.keys():
        model = alg_dict[alg_name]
        model.fit(X_train, Y_train)
        y_predict = model.predict(X_test)
        #score = -r2_score(Y_test, y_predict)
        #score = - mean_squared_error(Y_test, y_predict) ** 0.5
        mse = mean_squared_error(Y_test, y_predict)
        print(f"mse: {mse}")
        score = (-np.mean(cross_val_score(model, X, Y, cv=21, scoring='neg_mean_squared_error')))**0.5
        y_predict = cross_val_predict(model, X, Y, cv=21)
        print(f"{alg_name} {score}")
        if score < best_score:
            best_model = model
            best_score = score
    # save the best model
    print(f"best score {best_score} best model {best_model}")
    model_final = best_model
    # X_df = pd.DataFrame(feature_selection.transform(X), columns=best_features)
    # model_final.fit(X_df, Y)
    dataset['Y_predict'] = y_predict
    with plt.style.context([]):
        x = 'Y'
        y = 'Y_predict'
        r2 = r2_score(dataset[x], dataset[y])
        fig, ax = plt.subplots(dpi=400, figsize=(5, 5))
        plt.scatter(dataset[x], dataset[y], c='blue', alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        # x_data = result_df[x]
        # y_data = result_df[y]
        #
        lim_max = 6
        lim_min = 2.8
        plt.plot([lim_min, lim_max], [lim_min, lim_max], color='black', linestyle="--")
        # plt.xticks(fontsize=12, fontweight='bold')
        # plt.yticks(fontsize=12, fontweight='bold')
        # plt.xlabel(x, fontsize=12, fontweight='bold')
        # plt.ylabel(y, fontsize=12, fontweight='bold')
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.text(0.05, 0.95, "$R^2={r2}$".format(r2=round(r2, 2)), transform=ax.transAxes)
        plt.savefig(f'./figures/R2.png', bbox_inches='tight')