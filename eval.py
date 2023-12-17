"""
util functions for model evaluation
"""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import math
import sys
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from scipy.stats import pearsonr


def cal_reg_metric(y_true, y_predict):
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
    return dict({"n_samples": n, "MSE": MSE, "RMSE": RMSE, "MAE": MAE, "R2": R2, 'R': pccs,
                 })


def cal_cls_metric(y_true, y_pred):
    """
    Evaluate a classification model using different metrics including confusion matrix.
    :param y_true: True labels of the data.
    :param y_pred: Predicted labels of the data.
    :return: A dictionary containing different evaluation metrics.
    """
    conf_mat = confusion_matrix(y_true, y_pred)
    metrics = {
        "Confusion Matrix": conf_mat.tolist(),  # to_list make json
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }
    return metrics


def fcv(original_model, X, y, minimum_ratio=0.1, maximum_ratio=0.95, reverse=False, details=False, lite=False, k=10,
        m=1):
    """
    FCV method
    :param original_model:
    :param X:
    :param y:
    :param minimum_ratio:
    :param maximum_ratio:
    :param reverse: True for use low Y data for training
    :param details: If return the details for each fcv folder prediction
    :param lite:
    :param k: number of fcv folder
    :param m:
    :return:
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    if not reverse:
        arr1inds = y.argsort()
    else:
        arr1inds = y.argsort()[::-1]
    X = X[arr1inds]
    y = y[arr1inds]

    if not maximum_ratio:
        maximum_ratio = 1 - 1 / k
    sample_number = len(X)
    fold_sample_number = math.floor(sample_number / k)
    minimum_sample_number = round(minimum_ratio * sample_number)
    maxmum_sample_number = round(maximum_ratio * sample_number)

    label = []
    prediction = []
    max_value = []
    details = {
        "y_trains": [],
        "y_train_predicts": [],
        "y_tests": [],
        "y_test_predicts": [],
    }
    for split in range(fold_sample_number, sample_number, fold_sample_number if not lite else fold_sample_number * 10):
        if split < minimum_sample_number:
            continue
        if split > maxmum_sample_number:
            break

        #         print("Training 0 to {}, validation on {} to {}".format(split, split, split + fold_sample_number))
        sys.stdout.write("Training end in %s out of %s \r" % (split, sample_number))
        sys.stdout.flush()
        X_train = X[0:split]
        y_train = y[0:split]
        start_sample_number = split + (m - 1) * fold_sample_number
        end_sample_number = split + m * fold_sample_number

        if start_sample_number > sample_number:
            break
        else:
            end_sample_number = min(end_sample_number, sample_number)

        X_val = X[start_sample_number:end_sample_number]
        y_val = y[start_sample_number:end_sample_number]

        if issubclass(type(original_model), BaseEstimator):
            model = clone(original_model)
            model.fit(X_train, y_train)
        else:
            pass
            # if not hybrid:
            #     model = original_model(shape)
            #     model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=0)
            # else:
            #     model = hybrid_train(original_model, X_train, y_train, shape)
        y_pred = model.predict(X_val)

        label.extend(list(y_val))
        prediction.extend(list(y_pred))
        max_value.extend([y[split - 1]] * len(y_val))
        if details:
            y_train_predict = model.predict(X_train)
            details['y_trains'].append(y_train)
            details['y_train_predicts'].append(y_train_predict)
            details['y_tests'].append(y_val)
            details['y_test_predicts'].append(y_pred)

        # if not issubclass(type(original_model), BaseEstimator):
        #     K.clear_session()
    if details:
        return label, prediction, details
    else:
        return label, prediction


def cv(original_model, X, y, k=5):
    """ cross validation method
    :return y_true,  y_predict
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    if issubclass(type(original_model), BaseEstimator):     # sklearn model
        cv_prediction = cross_val_predict(original_model, X, y, cv=KFold(k, shuffle=True))
    else:
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # define 10-fold cross validation test harness
        kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

        y_random = []
        cv_prediction = []
        for i, (train, test) in enumerate(kfold.split(X, y)):
            sys.stdout.write('{} of {} fold \r'.format(i + 1, k))
            sys.stdout.flush()

            # train sklearn model
            if issubclass(type(original_model), BaseEstimator):
                model = clone(original_model)
                model.fit(X[train], y[train])
            else:
                model = clone(original_model)

            y_random.extend(y[test])
            cv_prediction.extend(model.predict(X[test]))

        y = y_random

    return y, cv_prediction


def forward_holdout_split(X, y, test_ratio, reverse=False):
    """
    :param X:
    :param y:
    :param test_ratio:
    :param reverse: False means y with high value as test set
    :return:
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if not reverse:
        arr1inds = y.argsort()
    else:
        arr1inds = y.argsort()[::-1]
    X = X[arr1inds]
    y = y[arr1inds]
    sample_number = len(X)
    split = round((1 - test_ratio) * sample_number)
    X_train = X[0:split]
    y_train = y[0:split]
    X_test = X[split:]
    y_test = y[split:]
    return X_train, y_train, X_test, y_test


def forward_holdout(original_model, X, y, test_ratio, reverse=False):
    """
    forward holdout method FH
    :param original_model:
    :param X:
    :param y:
    :param test_ratio:
    :param reverse:
    :return:
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    if not reverse:
        arr1inds = y.argsort()
    else:
        arr1inds = y.argsort()[::-1]  # for higher
    X = X[arr1inds]
    y = y[arr1inds]
    sample_number = len(X)
    split = round((1 - test_ratio) * sample_number)
    X_train = X[0:split]
    y_train = y[0:split]
    X_val = X[split:]
    y_val = y[split:]
    if issubclass(type(original_model), BaseEstimator):
        model = clone(original_model)
        model.fit(X_train, y_train)
    else:
        pass
        # if not hybrid:
        #     model = original_model(shape)
        #     model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=0)
        # else:
        #     model = hybrid_train(original_model, X_train, y_train, shape)
    y_pred = model.predict(X_val)
    return y_val, y_pred, X_train, y_train


def discovery_precision(y_extra_predict, y_inter_predict, y_extra_true, y_inter_true, alpha=None):
    """
    calculation of discovery_precision DP
    :param y_extra_predict: y prediction in validation set
    :param y_inter_predict: y out of bag prediction  in training set
    :param y_extra_true: y true in validation set
    :param y_inter_true: y true in training set
    :param alpha: top percent of predicted value scale:[0,1]
                  default set to test ratio  = len(Validation set)/ len(Total set)
    :return: score of discovery precision
    """
    if alpha is None:
        alpha = len(y_extra_true) / (len(y_extra_true) + len(y_inter_true))
    percent = alpha * 100
    all_true = np.array(list(y_extra_true) + list(y_inter_true))
    all_predict = np.array(list(y_extra_predict) + list(y_inter_predict))
    y_true_limit = np.percentile(all_true, 100 - percent)
    y_predict_limit = np.percentile(all_predict, 100 - percent)
    count = 0
    for i in range(len(all_true)):
        if all_true[i] >= y_true_limit and all_predict[i] >= y_predict_limit:
            count = count + 1
    n_total = len([i for i in all_predict if i >= y_predict_limit])
    score = count / n_total  # top FOM discovery probability
    return score


def score_dp_by_forward_holdout(model, X_train, Y_train, alpha=None,
                                test_ratio=0.1, reverse=True, cv_fold=5):
    """
    :param model:
    :param X_train:
    :param Y_train:
    :param alpha:
    :param test_ratio: test_ratio for FH method
    :param reverse:
    :param cv_fold: out of bag prediction cv folder
    :return: score_dp
    """
    y_true, y_predict, X_train_h, y_train_h = forward_holdout(model, X_train, Y_train,
                                                              test_ratio=test_ratio,
                                                              reverse=reverse)
    # get out of bag prediction by cv prediction
    y_inter_true, y_train_predict = cv(model, X_train_h, y_train_h, k=cv_fold)
    y_extra_predict = y_predict
    y_inter_predict = y_train_predict
    y_extra_true = y_true
    y_inter_true = y_inter_true
    score = discovery_precision(y_extra_predict, y_inter_predict, y_extra_true, y_inter_true, alpha=alpha)
    return score, y_extra_predict, y_inter_predict, y_extra_true, y_inter_true
