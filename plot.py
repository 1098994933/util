"""
functions of plot figures
"""
import warnings
from typing import List

from matplotlib.pylab import mpl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
        x_label (str): Label for the x-axis. Default is "Measured".
        y_label (str): Label for the y-axis. Default is "Predicted".
        title (str, optional): Title for the plot. Default is None.
        figure_size (tuple): Size of the figure. Default is (7, 5).
        alpha_test (float): Transparency for test set points. Default is 0.4.
        alpha_train (float): Transparency for train set points. Default is 0.4.
        legend_loc (str): Location of the legend. Default is 'best'.
        grid_style (str): Line style for the grid. Default is "--".
        save_path (str, optional): Path to save the plot. Default is None.
        plot_test_index (bool): Whether to label each point with its index. Default is False.
        evaluation_matrix (dict, optional): Dictionary containing evaluation metrics such as R2, MAE, and R.
    """

    # Calculate plot limits
    all_values = [Y_test, y_test_predict]
    if Y_train is not None and y_train_predict is not None:
        all_values.extend([Y_train, y_train_predict])

    # 计算全局最小值和最大值
    global_min = min(min(v) for v in all_values)
    global_max = max(max(v) for v in all_values)
    data_range = global_max - global_min

    # 处理数据范围为零或极小的情况
    if data_range == 0:
        if global_max == 0:
            padding = 1.0  # 如果全为0，默认扩展1个单位
        else:
            padding = abs(global_max) * 0.02  # 按绝对值的2%扩展
        lim_min = global_min - padding
        lim_max = global_max + padding
    else:
        padding = data_range * 0.02  # 扩展数据范围的2%
        lim_min = global_min - padding
        lim_max = global_max + padding

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
        plt.text(0.05, 0.70, '\n'.join(text_lines), transform=ax.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.6))
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


def plot_corr(dataset: pd.DataFrame, targets, save_path=None, title=None, figsize=None):
    """
    绘制特征相关性热力图

    参数：
    dataset : pd.DataFrame
    targets : list - 需要分析的特征列名列表
    save_path : str - 图片保存路径（可选）
    title : str - 图表标题（可选）
    figsize : tuple - 自定义图形尺寸（可选）
    annot_font_scale : float - 注释字体缩放系数（默认1.0）
    """
    if not isinstance(targets, list) or len(targets) < 2:
        raise ValueError("targets应为包含至少两个特征名的列表")

    corr_matrix = dataset[targets].corr()
    n_features = len(targets)
    # 动态计算图形尺寸
    base_size = 1.2  # 每个特征的基础尺寸单位
    min_size = 8  # 最小图形尺寸
    if figsize is None:
        # 计算自适应尺寸
        plot_size = max(min_size, base_size * n_features)
        figsize = (plot_size, plot_size * 0.8)

    label_fontsize = max(12, 18 - n_features // 3)
    # annot_size = max(8, 12 - n_features//5) * annot_font_scale

    # 创建画布
    plt.figure(figsize=figsize, dpi=300)
    with sns.axes_style("white"):
        ax = sns.heatmap(corr_matrix, annot=True, annot_kws={"size": 10, "weight": "bold"}, fmt=".2f",
                         cmap="rainbow", linewidths=0, vmin=-1, vmax=1, square=True,
                         cbar_kws={"shrink": 0.8}
                         )
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        # 调整坐标轴
        ax.tick_params(
            axis='both',
            which='both',
            labelsize=label_fontsize,
            rotation=45,
            labelrotation=45
        )

        # 设置标签对齐方式
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), ha="right", rotation_mode="anchor")
        # 添加标题
        if title:
            plt.title(
                title,
                fontsize=label_fontsize + 4,
                fontweight="bold",
                pad=20
            )

        # 调整颜色条
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=label_fontsize)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    return corr_matrix


from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


def plot_feature_importance(
        features: list,
        feature_importance: np.ndarray,
        n: int = 10,
        save_path: str = None,
        figsize: tuple = None,
        color_map: str = 'viridis',
        show_values: bool = True
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    可视化特征重要性并返回绘图数据

    参数：
    features : list - 特征名称列表
    feature_importance : np.ndarray - 特征重要性值数组
    n : int - 显示前n个重要特征 (默认10)
    save_path : str - 图片保存路径 (可选)
    figsize : tuple - 自定义图形尺寸 (可选)
    color_map : str - 颜色映射方案 (默认'viridis')
    show_values : bool - 是否显示重要性数值 (默认True)

    返回：
    Tuple[pd.DataFrame, plt.Figure] - (包含特征和重要性值的DataFrame, 图形对象)
    """

    # 参数验证
    if len(features) != len(feature_importance):
        raise ValueError("特征名称与重要性值数量不一致")
    if n <= 0:
        raise ValueError("n必须大于0")
    if len(features) == 0:
        raise ValueError("特征列表不能为空")

    # 数据预处理
    n = min(n, len(features))
    max_importance = feature_importance.max()

    # 处理全零重要性特殊情况
    if max_importance == 0:
        normalized_importance = np.zeros_like(feature_importance)
        warnings.warn("所有特征重要性为零，请检查模型", UserWarning)
    else:
        normalized_importance = 100.0 * (feature_importance / max_importance)

    # 排序处理
    sorted_idx = np.argsort(normalized_importance)[-n:][::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = normalized_importance[sorted_idx]

    # 创建返回数据
    result_df = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_importance,
        'raw_importance': feature_importance[sorted_idx]  # 保留原始值
    }).sort_values('importance', ascending=False)

    # 可视化部分
    fig = plt.figure(figsize=figsize or (10, max(4, 0.6 * n)), dpi=100)
    ax = fig.add_subplot(111)

    # 绘制条形图
    colors = plt.cm.get_cmap(color_map)(np.linspace(0.3, 1, n))
    bars = ax.barh(
        y=np.arange(n),
        width=sorted_importance,
        height=0.8,
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )

    # 添加数值标签
    if show_values:
        for bar in bars:
            width = bar.get_width()
            label = f"{width:.1f}%" if max_importance != 0 else "0.0%"
            ax.text(
                x=width + 0.5,
                y=bar.get_y() + bar.get_height() / 2,
                s=label,
                va='center',
                ha='left',
                fontsize=9,
                color='black'
            )

    # 坐标轴设置
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.invert_yaxis()

    # 样式优化
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#404040')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    ax.set_xlabel('Relative Importance (%)', fontsize=12)
    ax.set_title('Feature Importance Ranking', fontsize=14, pad=20)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    else:
        plt.tight_layout()
    return result_df, fig

import shap
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
