import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]


class SHAPExplainer(object):
    def __init__(self, model):
        self.model = model

    def generate_shap_figures(self, X, fig_path=f'./figures/shap.png', n_features=5):
        """
        draw and save figures
        :param n_features: number of top features to include in the dependent plot
        :param model: be fitted tree based model
        :param X: DataFrame
        :param fig_path:
        :return: top n feature names
        """
        model = self.model
        n_features = min(X.shape[1], n_features)
        if type(model).__name__ in ["GradientBoostingRegressor", "RandomForestRegressor"]:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        else:
            explainer = shap.KernelExplainer(model.predict, X, **{"keep_index": True})  # bug fix
        shap_values = explainer.shap_values(X)
        # summary_plot
        plt.figure(dpi=300)
        shap.summary_plot(shap_values, X, show=False, color_bar=True)
        plt.xlabel("SHAP value of model", fontweight='bold', fontsize=24)
        plt.tick_params(labelsize=24, )
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        # dependent plots
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)

        # Sort features by importance and select the top n
        top_features_indices = np.argsort(feature_importance)[-n_features:]
        top_features_names = X.columns[top_features_indices]

        # Generate dependent plots for top features
        for feature_name in top_features_names:
            plt.figure(dpi=300)
            shap.dependence_plot(feature_name, shap_values, X, show=False)
            # Save the dependent plots
            plt.savefig(f"{fig_path}_{feature_name}.png", bbox_inches='tight')
            plt.close()
        plt.close('all')
        return top_features_names
