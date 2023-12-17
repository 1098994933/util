import pandas as pd


def get_magpie_features(file_name="train_formula.csv", data_path="/content/"):
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers import composition as cf
    # magpie
    df_chemistry_formula = pd.read_csv(data_path + file_name)
    df_magpie = StrToComposition(target_col_id='composition_obj').featurize_dataframe(df_chemistry_formula, 'formula')
    feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                              cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
    feature_labels = feature_calculators.feature_labels()
    df_magpie = feature_calculators.featurize_dataframe(df_magpie, col_id='composition_obj');
    df_magpie.to_csv(data_path + f'{file_name}_magpie_features.csv', index=False)
