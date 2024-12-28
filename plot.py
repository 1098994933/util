"""
functions of plot figures
"""

import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号


def plot_regression_results(Y_test, y_test_predict, Y_train=None, y_train_predict=None,
                            x_label="Measured", y_label="Predicted", title=None,
                            figure_size=(7, 5), alpha_test=0.4, alpha_train=0.4,
                            legend_loc='best', grid_style="--", save_path=None,
                            plot_test_index=False, evaluation_matrix=None):
    """
    Plot regression results comparing measured and predicted values.

    Parameters:
        Y_test (list or array): Actual values for the test set.
        y_test_predict (list or array): Predicted values for the test set.
        Y_train (list or array, optional): Actual values for the training set. Default is None.
        y_train_predict (list or array, optional): Predicted values for the training set. Default is None.
        x_label (str): Label for the x-axis. Default is "Measured(CFS)".
        y_label (str): Label for the y-axis. Default is "Predicted(CFS)".
        title (str, optional): Title for the plot. Default is None.
        figure_size (tuple): Size of the figure. Default is (7, 5).
        font (str): Font for text. Default is 'Arial'.
        alpha_test (float): Transparency for test set points. Default is 0.4.
        alpha_train (float): Transparency for train set points. Default is 0.4.
        legend_loc (str): Location of the legend. Default is 'best'.
        grid_style (str): Line style for the grid. Default is "--".
        save_path (str, optional): Path to save the plot. Default is None.
        plot_test_index (bool): Whether to label each point with its index. Default is False.
        evaluation_matrix (dict, optional): Dictionary containing evaluation metrics such as R2, MAE, and R. Default is None.
    """

    # Calculate plot limits
    all_values = [Y_test, y_test_predict]
    if Y_train is not None and y_train_predict is not None:
        all_values.extend([Y_train, y_train_predict])

    lim_max = max(max(v) for v in all_values) * 1.02
    lim_min = min(min(v) for v in all_values) * 0.98

    # Plot setup
    plt.figure(figsize=figure_size)
    plt.grid(linestyle=grid_style)
    # Scatter plots
    plt.scatter(Y_test, y_test_predict, color='red', alpha=alpha_test, label='Test')
    if Y_train is not None and y_train_predict is not None:
        plt.scatter(Y_train, y_train_predict, color='blue', alpha=alpha_train, label='Train')

    # Plot y = x line
    plt.plot([lim_min, lim_max], [lim_min, lim_max], color='blue')

    # Set axis properties
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)

    ax = plt.gca()
    # Add evaluation metrics as text on the plot
    if evaluation_matrix:
        r2 = evaluation_matrix.get("R2", None)
        mae = evaluation_matrix.get("MAE", None)
        r = evaluation_matrix.get("R", None)
        text_lines = []
        if r2 is not None:
            text_lines.append(f"$R^2={r2:.3f}$")
        if mae is not None:
            text_lines.append(f"$MAE={mae:.3f}$")
        if r is not None:
            text_lines.append(f"$R={r:.3f}$")
        plt.text(0.05, 0.70, '\n'.join(text_lines), transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
    # Add optional title
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    # Add legend
    if Y_train is not None and y_train_predict is not None:
        plt.legend(loc=legend_loc)
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Optionally label each point with its index
    if plot_test_index:
        for i, (x, y) in enumerate(zip(Y_test, y_test_predict)):
            plt.text(x, y, f'{i}', fontsize=8, ha='right', va='bottom', color='black')

    # Show plot
    plt.show()

import shap
import matplotlib.pyplot as plt
import numpy as np


def generate_shap_figures(model, X, fig_path='./figures/shap.png', n_features=5):
    """
    Generate SHAP summary and dependence plots, and save them to files.

    Parameters:
        model: Trained tree-based model or compatible model with a predict function.
        X (pd.DataFrame): Input data for SHAP analysis.
        fig_path (str): Base path to save SHAP figures.
        n_features (int): Number of top features to include in dependence plots.

    Returns:
        list: Top `n_features` feature names.
    """
    # Limit the number of features to the number of columns in X
    n_features = min(X.shape[1], n_features)

    # Determine the appropriate SHAP explainer
    if type(model).__name__ in ["GradientBoostingRegressor", "RandomForestRegressor"]:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    else:
        explainer = shap.KernelExplainer(model.predict, X, keep_index=True)

    # Compute SHAP values
    shap_values = explainer.shap_values(X)

    # Generate and save summary plot
    plt.figure(dpi=300)
    shap.summary_plot(shap_values, X, show=False, color_bar=True)
    plt.xlabel("SHAP value of model", fontweight='bold', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    # Compute feature importance and select top features
    if n_features <= 0:
        return []
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_features_indices = np.argsort(feature_importance)[-n_features:]
    top_features_names = X.columns[top_features_indices]

    # Generate and save dependence plots for top features
    for feature_name in top_features_names:
        plt.figure(dpi=300)
        shap.dependence_plot(feature_name, shap_values, X, show=False)
        plt.savefig(f"{fig_path}_{feature_name}.png", bbox_inches='tight')
        plt.close()

    # Close all remaining plots to release resources
    plt.close('all')
    return top_features_names
