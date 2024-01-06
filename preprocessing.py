from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocessing_dataset(df):
    """
    :return:
    """
    features = list(df.columns)
    scale = MinMaxScaler()
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(df)
    X = pd.DataFrame(scale.fit_transform(X), columns=features)
    return X, scale, imp