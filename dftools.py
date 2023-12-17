import pandas as pd
import numpy as np
import string
from pathlib import Path


def round_series(series: pd.Series, decimal: int = 3) -> pd.Series:
    if series.dtype == object:
        return series
    else:
        return series.round(decimal)


def round_data(dataframe: pd.DataFrame, decimal: int = 3) -> pd.DataFrame:
    """
    round data in DataFrame
    """
    return dataframe.apply(round_series, args=[decimal])


def secure_filename(_string, addition=[]):
    path = Path(_string)
    filename = path.stem
    string_punctuation = string.punctuation
    string_punctuation = string_punctuation.translate(str.maketrans("", "", "."))

    if len(addition) == 0:
        translation_str = string_punctuation
    else:
        translation_str = string_punctuation + "".join(addition)

    translate_table = str.maketrans("", "", translation_str)

    return filename.translate(translate_table)
