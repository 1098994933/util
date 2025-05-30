from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

default_search_space = {
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
            'max_depth': hp.choice('max_depth', range(1, 5)),
            'max_features': hp.choice('max_features', [100, 1000, 3000]),
            'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
            'criterion': hp.choice('criterion',
                                   ["squared_error", "absolute_error"])}
    }

}


def opt_hyper(opt_function, space4rf):
    best = 0

    def f(params):
        global best
        acc = opt_function(params)
        if acc > best:
            best = acc
        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(f, space4rf, algo=tpe.suggest, max_evals=300, trials=trials)
    print('best:')
    print(best)
    return best


if __name__ == '__main__':
    search_space = {
        'max_depth': hp.choice('max_depth', range(1, 50)),
        'max_features': hp.choice('max_features', [100, 1000, 3000]),
        'n_estimators': hp.choice('n_estimators', [100, 1000, 3000]),
        'criterion': hp.choice('criterion',
                               ["squared_error", "absolute_error"])}

    X = None
    Y = None


    def opt_function(params):
        X_ = X[:]
        model = RandomForestRegressor(**params)
        return cross_val_score(model, X, Y).mean()


    opt_hyper(opt_function, search_space)
