"""
regressor
"""
import json
import os
import pickle
import copy
import uuid
from typing import List

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from Config import Config
from eval import cal_reg_metric, cv, score_dp_by_forward_holdout
from feature_explainer.SHAPExplainer import SHAPExplainer

def lasso(**param):
    return make_pipeline(StandardScaler(with_mean=False), Lasso(**param))


def ridge(**param):
    return make_pipeline(StandardScaler(with_mean=False), Ridge(**param))


def svr(**param):
    return make_pipeline(StandardScaler(with_mean=False), SVR(**param))


# TODO parameter limitation
# TODO hyper parameter opt scope
reg_model_config = {
    "RandomForestRegressor": {
        "model": RandomForestRegressor,
        "model_params": {
            "n_estimators": 100,
            "min_samples_split": 2,
            "random_state": 0
        },
        "model_params_type": {
            "n_estimators": "int",
            "min_samples_split": "int",
            "random_state": "int"
        },
        "hyper_param_search": {
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": [0.5, 0.1, 0.05, 0.01],
        },
    },
    "GradientBoostingRegressor": {
        "model": GradientBoostingRegressor,
        "model_params": {
            "n_estimators": 100,
            "max_depth": 3,
            "random_state": 0
        },
        "model_params_type": {
            "n_estimators": "int",
            "max_depth": "int",
            "random_state": "int"
        },
        "hyper_param_search": {
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": [0.5, 0.1, 0.05, 0.01]
        }
    },
    "LinearRegression": {
        "model": LinearRegression,
        "model_params": {
        },
        "model_params_type": {
        },
        "hyper_param_search": {
        }
    },

    "Lasso": {
        "model": lasso,
        "model_params": {
            "alpha": 1
        },
        "model_params_type": {
            "alpha": "float"
        },
        "hyper_param_search": {
            "alpha": [0.1, 1.0, 10.0]
        }
    },

    "Ridge": {
        "model": ridge,
        "model_params": {
            "alpha": 1
        },
        "model_params_type": {
            "alpha": "float"
        },
        "hyper_param_search": {
            "alpha": [0.1, 1.0, 10.0]
        }
    },

    "SVR": {
        "model": svr,
        "model_params": {
            "C": 1,
            "kernel": "rbf",
        },
        "model_params_type": {
            "C": "float",
            "kernel": "list"
        },
        "hyper_param_search": {
            "C": [0.1, 1.0, 10.0, 100],
            "kernel": ["linear", "rbf"]
        }
    },
    "MLPRegressor": {
        "model": MLPRegressor,
        "model_params": {
            "solver": "adam",
            "max_iter": 200,
            "random_state": 0
        },
        "model_params_type": {
            "solver": "list",
            "max_iter": "list",
            "random_state": "int"
        },
        "hyper_param_search": {
            "solver": ["adam", "sgd"],
            "max_iter": [200, 300, 500]
        }
    },
    "XGBoost": {
        "model": xgb.XGBRegressor,
        "model_params": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 0
        },
        "model_params_type": {
            "n_estimators": "int",
            "max_depth": "int",
            "learning_rate": "float",
            "random_state": "int"
        },
        "hyper_param_search": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.1, 0.01, 0.001]
        }
    }
}


class RegressionModel(object):
    def __init__(self, x_train, y_train, x_test=None, y_test=None, model_id: str = None, model_path="./"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        if model_id is None:
            self.model_id = str(uuid.uuid4())  # for model save path and identification
        else:
            self.model_id = model_id

        self.model = None
        self.best_model = None
        self.best_model_info = None
        self.model_path = model_path
        self.cv_fold = min(5, len(self.y_train))
    @property
    def reg_alg_config(self):
        return reg_model_config

    def fit_reg_model(self, alg_name="RandomForestRegressor", param=None):
        alg = reg_model_config[alg_name]["model"]
        if param is None:  # use default param
            params = reg_model_config[alg_name]["model_params"]
        else:
            params = reg_model_config[alg_name]["model_params"].copy()
            for key in param.keys():
                params[key] = param[key]
        reg_model = alg(**params)
        reg_model.fit_cls_model(self.x_train, self.y_train)
        self.model = reg_model
        train_y_predict = list(np.array(self.model.predict(self.x_train), dtype=float))
        train_y_true = list(np.array(self.y_train, dtype=float))
        train_metrics_dict = cal_reg_metric(train_y_true, train_y_predict)
        train_metrics_dict['discovery_precision'] = ""  # placeholder

        # Save the trained model to disk
        with open(os.path.join(self.model_path, f"{self.model_id}.pickle"), "wb") as file:
            pickle.dump(self.model, file)
        # Serialize the model parameters as JSON
        model_params_json = json.dumps(params)
        # cv
        cv_y_true, cv_y_predict = cv(self.model, self.x_train, self.y_train, k=self.cv_fold)

        cv_y_true = list(np.array(cv_y_true, dtype=float))  # make it serializable
        cv_y_predict = list(np.array(cv_y_predict, dtype=float))
        discovery_precision = score_dp_by_forward_holdout(self.model, self.x_train, self.y_train, alpha=None,
                                                          test_ratio=0.1, reverse=True, cv_fold=self.cv_fold)
        cv_metrics_dict = cal_reg_metric(cv_y_true, cv_y_predict)
        cv_metrics_dict['discovery_precision'] = discovery_precision

        model_info = {
            "model_id": self.model_id,
            "task_type": "reg",
            "alg": alg_name,
            "eval": {
                "train": {
                    "y_true": train_y_true,
                    "y_predict": train_y_predict,
                    "metrics": train_metrics_dict
                },
                "cv": {
                    "y_true": cv_y_true,
                    "y_predict": cv_y_predict,
                    "metrics": cv_metrics_dict
                },
            },
            "param": model_params_json,
        }
        # model test if x_test is not None
        if self.x_test is not None:
            test_y_predict = list(np.array(self.model.predict(self.x_test), dtype=float))
            test_y_true = list(np.array(self.y_test, dtype=float))
            test_metrics_dict = cal_reg_metric(test_y_true, test_y_predict)
            model_info["eval"]["test"] = {}
            model_info["eval"]["test"]["y_true"] = test_y_true
            model_info["eval"]["test"]["y_predict"] = test_y_predict
            model_info["eval"]["test"]["metrics"] = test_metrics_dict
        return reg_model, model_info

    def fit_best_reg_model(self, model_id, shap=True):
        best_model_info = None
        best_score = -np.inf
        best_model = None
        self.x_train.columns = self.x_train.columns.astype(str)
        for alg_name in reg_model_config.keys():
            alg = reg_model_config[alg_name]["model"]
            params = reg_model_config[alg_name]["model_params"]  # use default param
            reg_model = alg(**params)
            reg_model.fit_cls_model(self.x_train, self.y_train)
            model = reg_model
            train_y_predict = list(np.array(model.predict(self.x_train), dtype=float))
            train_y_true = list(np.array(self.y_train, dtype=float))
            train_metrics_dict = cal_reg_metric(train_y_true, train_y_predict)
            train_metrics_dict['discovery_precision'] = ""  # placeholder
            # Serialize the model parameters as JSON
            model_params_json = json.dumps(params)
            # cv
            cv_y_true, cv_y_predict = cv(model, self.x_train, self.y_train, k=self.cv_fold)
            cv_y_true = list(np.array(cv_y_true, dtype=float))  # make it serializable
            cv_y_predict = list(np.array(cv_y_predict, dtype=float))
            cv_metrics_dict = cal_reg_metric(cv_y_true, cv_y_predict)
            if best_score < cv_metrics_dict["R2"]:  # a better model
                best_score = cv_metrics_dict["R2"]
                best_model = model
            else:
                continue
            # discovery_precision = score_dp_by_forward_holdout(model, self.x_train, self.y_train, alpha=None,
            #                                                   test_ratio=0.1, reverse=True, cv_fold=self.cv_fold)
            #
            # cv_metrics_dict['discovery_precision'] = discovery_precision

            model_info = {
                "model_id": model_id,
                "task_type": "reg",
                "alg": alg_name,
                "eval": {
                    "train": {
                        "y_true": train_y_true,
                        "y_predict": train_y_predict,
                        "metrics": train_metrics_dict
                    },
                    "cv": {
                        "y_true": cv_y_true,
                        "y_predict": cv_y_predict,
                        "metrics": cv_metrics_dict
                    },
                },
                "param": model_params_json,
            }
            best_model_info = model_info

        # Save the trained model with best performance
        with open(os.path.join(self.model_path, f"{model_id}.pickle"), "wb") as file:
            pickle.dump(best_model, file)
        if shap:
            fig_path = os.path.join(self.model_path, f"{model_id}_shap_value.png")
            explainer = SHAPExplainer(best_model)
            top_features_names = explainer.generate_shap_figures(X=self.x_train, fig_path=fig_path)
            best_model_info["shap_summary_plot"] = f"/get_shap_summary/{model_id}"
            best_model_info["shap_dependent_plot"] = list(top_features_names)

        return best_model, best_model_info

    def fit_reg_models(self, alg_names: List[str], alg_params):
        """
        :param alg_names: List of alg_name
        :return:
        """
        model_infos = {}
        if alg_params is None:
            alg_params = [None] * len(alg_names)
        for alg_name, alg_param in zip(alg_names, alg_params):
            reg_model, model_info = self.fit_reg_model(alg_name, alg_param)
            model_infos[model_info["alg"]] = model_info
            if self.best_model_info is None:
                self.best_model_info = model_info
                self.best_model = reg_model
            elif model_info["eval"]["cv"]["metrics"]["RMSE"] < self.best_model_info["eval"]["cv"]["metrics"]["RMSE"]:
                self.best_model_info = model_info
                self.best_model = reg_model
        return model_infos

    def save_model(self, path: str):
        """

        """
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def load_model(self, path: str):
        with open(path, "rb") as file:
            reg_model = pickle.load(file)
        return reg_model

