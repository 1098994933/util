import pandas as pd
import os


def get_magpie_features(file_name="train_formula.csv", data_path=None, alloy_features=False) -> pd.DataFrame:
    """
    将含有化学式的csv文件，计算其magpie features remark: this function must run under in __main__:
    :param file_name: a csv file containing a column "formula"
    :param data_path: the directory path contain the file_name csv
    :param alloy_features: set True to add feature calculation of WenAlloys
    :return: pd.DataFrame with magpie features
    """
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers import composition as cf
    from matminer.featurizers.composition.alloy import WenAlloys

    # magpie
    df_chemistry_formula = pd.read_csv(os.path.join(data_path, file_name))
    df_magpie = StrToComposition(target_col_id='composition_obj').featurize_dataframe(df_chemistry_formula, 'formula')
    if alloy_features:
        feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                                  cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True),
                                                  WenAlloys()])
    else:
        feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                                  cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    feature_labels = feature_calculators.feature_labels()
    df_magpie = feature_calculators.featurize_dataframe(df_magpie, col_id='composition_obj')
    return df_magpie
