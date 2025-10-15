"""
functions of plot figures
"""
from typing import List, Optional

from matplotlib.figure import Figure
from matplotlib.pylab import mpl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Tuple
import warnings

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号


def plot_regression_results(y_test, y_test_predict, y_train=None, y_train_predict=None,
                            x_label="Measured", y_label="Predicted", title=None,
                            figure_size=(7, 5), alpha_test=0.4, alpha_train=0.4,
                            legend_loc='best', grid_style="--", save_path=None,
                            plot_test_index=False, evaluation_matrix=None):
    """
    Plot regression results comparing measured and predicted values.

    Parameters:
        y_test (list or array): Actual values for the test set.
        y_test_predict (list or array): Predicted values for the test set.
        y_train (list or array, optional): Actual values for the training set. Default is None.
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
    all_values = [y_test, y_test_predict]
    if y_train is not None and y_train_predict is not None:
        all_values.extend([y_train, y_train_predict])

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
    plt.scatter(y_test, y_test_predict, color='red', alpha=alpha_test, label='Test')
    if y_train is not None and y_train_predict is not None:
        plt.scatter(y_train, y_train_predict, color='blue', alpha=alpha_train, label='Train')

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

        def format_eval_number(num):
            """根据数值大小自动选择显示格式"""
            if abs(num) < 1e-3 or abs(num) > 1e6:
                return f"{num:.2e}"
            elif abs(num) < 1:
                return f"{num:.6f}".rstrip('0').rstrip('.')
            else:
                return f"{num:.3f}"

        if r2 is not None:
            text_lines.append(f"$R^2={format_eval_number(r2)}$")
        if mae is not None:
            text_lines.append(f"$MAE={format_eval_number(mae)}$")
        if r is not None:
            text_lines.append(f"$R={format_eval_number(r)}$")
        plt.text(0.05, 0.70, '\n'.join(text_lines), transform=ax.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.6))
    # Add optional title
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    # Add legend
    if y_train is not None and y_train_predict is not None:
        plt.legend(loc=legend_loc)
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Optionally label each point with its index
    if plot_test_index:
        for i, (x, y) in enumerate(zip(y_test, y_test_predict)):
            plt.text(x, y, f'{i}', fontsize=8, ha='right', va='bottom', color='black')

    # Show plot
    return plt.gcf()


def plot_corr(dataset: pd.DataFrame, targets: list, save_path: Optional[str] = None, title: Optional[str] = None,
              figsize: Optional[tuple] = None) -> Figure:
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
    fig = plt.figure(figsize=figsize, dpi=300)
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
    return fig


def plot_feature_importance(
        features: list,
        feature_importance: np.ndarray,
        n: int = 10,
        save_path: Optional[str] = None,
        figsize: Optional[tuple] = None,
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


def plot_decision_tree(model, feature_names=None, class_names=None, save_path=None,
                       figsize=(20, 10), max_depth=None, fontsize=10,
                       filled=True, rounded=True, impurity=True):
    """
    可视化决策树模型

    params:
    model: sklearn.tree.DecisionTreeClassifier 或 DecisionTreeRegressor
        训练好的决策树模型
    feature_names: list, optional
        特征名称列表，如果为None则使用默认的X[0], X[1]等
    class_names: list, optional
        类别名称列表，仅用于分类问题
    save_path: str, optional
        图片保存路径
    figsize: tuple, optional
        图形大小，默认为(20, 10)
    max_depth: int, optional
        显示的最大深度，如果为None则显示完整树
    fontsize: int, optional
        字体大小，默认为10
    filled: bool, optional
        是否填充节点颜色，默认为True
    rounded: bool, optional
        是否使用圆角矩形，默认为True
    impurity: bool, optional
        是否显示不纯度，默认为True

    return:
    fig: matplotlib.figure.Figure
        图形对象
    """
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    # 创建图形
    fig = plt.figure(figsize=figsize)

    # 绘制决策树
    plot_tree(model,
              feature_names=feature_names,
              class_names=class_names,
              max_depth=max_depth,
              fontsize=fontsize,
              filled=filled,
              rounded=rounded,
              impurity=impurity)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def process_1d_vectors(x, y, z, resolution=50):
    """
    将x,y,z 三个 一维向量形式的数据转换为二维网格数据

    参数:
        x, y, z: 一维向量数据
        resolution: 生成网格的分辨率

    返回:
        X, Y: 二维网格坐标
        Z: 插值后的二维网格数据
    """
    # 创建网格
    xi = np.linspace(min(x), max(x), resolution)
    yi = np.linspace(min(y), max(y), resolution)
    X, Y = np.meshgrid(xi, yi)

    # 使用线性插值填充网格
    Z = griddata((x, y), z, (X, Y), method='linear')

    return X, Y, Z


def plot_grouped_contour(x, y, z, groups=None, resolution=50, levels=20, cmap="jet",
                         figsize=(15, 10), title="", share_colorbar=True, if_scatter=True):
    """
    contour by groups，所有子图共享相同的colorbar范围

    参数:
        x, y, z: 一维向量数据或pandas Series
        groups: 一维向量，指定每个数据点的分组
        resolution: 网格分辨率
        levels: 等高线数量
        cmap: 颜色映射
        figsize: 图表大小
        title: 总标题
        share_colorbar: 是否共享colorbar范围
    """

    # 获取Series的名称作为坐标轴标签
    x_label = x.name if isinstance(x, pd.Series) and x.name is not None else 'X轴'
    y_label = y.name if isinstance(y, pd.Series) and y.name is not None else 'Y轴'
    z_label = z.name if isinstance(z, pd.Series) and z.name is not None else 'Z值'

    # 转换为numpy数组进行处理
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    # 计算全局z范围
    z_min, z_max = np.nanmin(z), np.nanmax(z)
    if isinstance(levels, int):
        levels = np.linspace(z_min, z_max, levels)
    if groups is not None:
        groups = np.asarray(groups)
    else:
        groups = np.zeros(len(x))
    # 获取唯一的分组
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # 计算行列数，使子图布局尽量接近正方形
    n_cols = int(np.ceil(np.sqrt(n_groups)))
    n_rows = int(np.ceil(n_groups / n_cols))

    # 创建图形和子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True,
                             squeeze=False)

    # 如果需要共享colorbar，计算全局的Z值范围
    if share_colorbar:
        global_vmin = np.min(z)
        global_vmax = np.max(z)
    else:
        global_vmin = None
        global_vmax = None

    # 绘制每个分组的等高线图
    for i, group in enumerate(unique_groups):
        # 获取当前组的数据
        mask = groups == group
        group_x = x[mask]
        group_y = y[mask]
        group_z = z[mask]

        # 计算子图位置
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # 检查x, y, z是否为常数
        x_is_constant = max(group_x) - min(group_x) < 1e-20
        y_is_constant = max(group_y) - min(group_y) < 1e-20
        z_is_constant = max(group_z) - min(group_z) < 1e-20

        if x_is_constant or y_is_constant or z_is_constant:
            # 如果任一维度为常数，创建散点图
            scatter = ax.scatter(group_x, group_y, c=group_z,
                                 cmap=cmap,
                                 vmin=global_vmin if share_colorbar else min(group_z),
                                 vmax=global_vmax if share_colorbar else max(group_z))

            # 添加常数值标注
            def format_number(num):
                """根据数值大小自动选择显示格式"""
                if abs(num) < 1e-3 or abs(num) > 1e6:
                    # 对于很小的数或很大的数使用科学计数法
                    return f"{num:.2e}"
                elif abs(num) < 1:
                    # 对于小于1的数，保留更多小数位
                    return f"{num:.6f}".rstrip('0').rstrip('.')
                else:
                    # 对于正常范围的数，保留2位小数
                    return f"{num:.2f}"

            if x_is_constant:
                ax.text(0.02, 0.98, f'X = {format_number(group_x[0])}',
                        transform=ax.transAxes, va='top')
            if y_is_constant:
                ax.text(0.02, 0.93, f'Y = {format_number(group_y[0])}',
                        transform=ax.transAxes, va='top')
            if z_is_constant:
                ax.text(0.02, 0.88, f'Z = {format_number(group_z[0])}',
                        transform=ax.transAxes, va='top')

            contour = scatter  # 为了后面的colorbar
        else:
            # 转换为网格数据
            X, Y, Z = process_1d_vectors(group_x, group_y, group_z, resolution=resolution)

            # set vmin 和 vmax
            if share_colorbar:
                vmin, vmax = global_vmin, global_vmax
            else:
                vmin, vmax = np.min(group_z), np.max(group_z)

            # 绘制等高线图
            contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8,
                                  vmin=vmin, vmax=vmax)
            if if_scatter:
                # 空心散点图
                ax.scatter(group_x, group_y, s=10, alpha=0.3, facecolors='none', edgecolors='black')

        # 设置子图标题
        ax.set_title(f"{group}", fontsize=14)

        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

    # 隐藏空的子图
    for i in range(n_groups, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    # 添加总标题
    fig.suptitle(title, fontsize=18, y=0.95)

    # 添加共享的colorbar
    if share_colorbar:
        cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))  # [左, 下, 宽, 高]
        cbar = fig.colorbar(contour, cax=cbar_ax)
        cbar.set_label(z_label, fontsize=12)

    # 调整布局
    plt.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))  # 为colorbar留出空间

    return fig, axes


def plot_group_by_freq_mean(x: pd.Series, y: pd.Series, groups: Optional[pd.Series] = None, figsize=(15, 10),
                            title=None):
    """
    对x和y进行分组，并绘制，每个分组的x y 按照x的大小等频率分组，每个分组绘制一个，每个分组的x y

    参数:
        x: pd.Series - X轴数据
        y: pd.Series - Y轴数据
        groups: pd.Series, optional - 分组数据，如果为None则默认分为一组
        figsize: tuple - 图形尺寸
        title: str, optional - 图形标题
    """

    x_label = x.name if isinstance(x, pd.Series) and x.name is not None else 'X轴'
    y_label = y.name if isinstance(y, pd.Series) and y.name is not None else 'Y轴'

    # 如果groups为None，创建默认分组
    if groups is None:
        groups = pd.Series(['Default'] * len(x), name='分组')

    # 获取唯一的分组
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    # 计算行列数，使子图布局尽量接近正方形
    n_cols = int(np.ceil(np.sqrt(n_groups)))
    n_rows = int(np.ceil(n_groups / n_cols))

    # 创建图形和子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True,
                             squeeze=False)

    for i, group in enumerate(unique_groups):
        # 获取当前组的数据
        mask = groups == group
        group_x = x[mask]
        group_y = y[mask]

        # 计算子图位置
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # 按照x的大小等频率分组 使用pd.qcut label为x的中位数
        group_x_quantile = pd.qcut(group_x, 10, labels=False, duplicates='drop')

        # 收集所有散点坐标
        x_means = []
        y_means = []

        for quantile in sorted(group_x_quantile.unique()):
            mask = group_x_quantile == quantile
            x_mean = group_x[mask].mean()
            y_mean = group_y[mask].mean()
            x_means.append(x_mean)
            y_means.append(y_mean)

        # 按照X值排序，确保折线沿着X增大的方向
        sorted_points = sorted(zip(x_means, y_means), key=lambda x: x[0])
        sorted_x_points = [point[0] for point in sorted_points]
        sorted_y_points = [point[1] for point in sorted_points]

        # 绘制连接所有散点的折线
        if len(sorted_x_points) > 1:
            ax.plot(sorted_x_points, sorted_y_points, color='black', linewidth=2, marker='o')
        else:
            ax.scatter(sorted_x_points, sorted_y_points, color='black', s=50)
        ax.set_title(f"{group}")
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
    # 隐藏空的子图
    for i in range(n_groups, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    # 添加总标题
    if title:
        fig.suptitle(title, fontsize=18, y=0.95)
    plt.tight_layout()

    return fig, axes


def plot_group_time_rolling(time_col: pd.Series, x: pd.Series, y: pd.Series, groups: Optional[pd.Series] = None,
                            figsize=(15, 10), title=None, max_segments: int = 30):
    """
    对时间序列数据进行分组，并按时间顺序分段取平均值后绘制，提高大数据量时的绘图性能

    参数:
        time_col: pd.Series - 时间列数据，用于排序和分段
        x: pd.Series - X轴数据，将显示在次坐标轴上
        y: pd.Series - Y轴数据，将显示在主坐标轴上
        groups: pd.Series, optional - 分组数据，如果为None则默认分为一组
        figsize: tuple - 图形尺寸
        title: str, optional - 图形标题
        min_periods: int - 滚动窗口计算所需的最小观测数，默认为1
        max_segments: int - 每个分组最多分成的段数，默认为30
    """

    # 参数验证
    if len(time_col) != len(x) or len(time_col) != len(y):
        raise ValueError("时间列、x列和y列的长度必须相同")

    # label
    time_label = time_col.name if isinstance(time_col, pd.Series) and time_col.name is not None else '时间'
    x_label = x.name if isinstance(x, pd.Series) and x.name is not None else 'X值'
    y_label = y.name if isinstance(y, pd.Series) and y.name is not None else 'Y值'

    df = pd.DataFrame({'time': time_col, 'x': x, 'y': y})
    df = df.dropna()
    if len(df) == 0:
        raise ValueError("dataset has no valid data")
    # 如果groups为None，创建默认分组
    if groups is None:
        df['group'] = ' '
    else:
        df['group'] = groups

    # 获取唯一的分组
    unique_groups = df['group'].unique()
    n_groups = len(unique_groups)

    # 排除数据量小于2的组
    unique_groups = [group for group in unique_groups if len(df[df['group'] == group]) > 1]
    n_groups = len(unique_groups)

    # 计算行列数，使子图布局尽量接近正方形
    n_cols = int(np.ceil(np.sqrt(n_groups)))
    n_rows = int(np.ceil(n_groups / n_cols))

    # 预处理：按组排序并计算分段数据
    def compute_segment_data_vectorized(group_data, max_segments):
        """使用完全向量化的numpy操作计算分段数据"""
        n_data = len(group_data)
        if n_data <= 1:
            return np.array([]), np.array([]), np.array([])

        n_segments = min(max_segments, n_data)  # number of time segment
        segment_indices = np.linspace(0, n_data - 1, n_segments, dtype=int)

        # 转换为numpy数组以提高性能
        time_values = group_data['time'].values
        x_values = group_data['x'].values
        y_values = group_data['y'].values

        # 完全向量化的分段计算
        n_segments_actual = len(segment_indices) - 1
        if n_segments_actual == 0:
            return np.array([]), np.array([]), np.array([])

        # 创建分段索引
        start_indices = segment_indices[:-1]
        end_indices = segment_indices[1:]

        # 使用numpy的高级索引和广播进行完全向量化计算
        # plot data ponits
        segment_times = []
        segment_x_means = np.zeros(n_segments_actual)
        segment_y_means = np.zeros(n_segments_actual)

        # 检查时间列是否为字符串类型
        is_time_string = pd.api.types.is_string_dtype(time_values) or (len(time_values) > 0 and isinstance(time_values[0], str))

        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            # 处理时间列：根据数据类型选择合适的方法
            if is_time_string:
                segment_time = group_data['time'].iloc[start:end+1].min()
            else:
                segment_time = np.min(time_values[start:end+1])

            segment_times.append(segment_time)
            segment_x_means[i] = np.mean(x_values[start:end+1])
            segment_y_means[i] = np.mean(y_values[start:end+1])

        segment_times = np.array(segment_times)

        return segment_times, segment_x_means, segment_y_means

    # 一次性计算所有组的分段数据，避免重复计算
    group_segment_data = {}
    all_x_values = []
    all_y_values = []

    # 预先排序所有数据，避免在循环中重复排序
    df_sorted = df.sort_values(by=['group', 'time'])

    # 使用groupby进行高效分组，避免重复的布尔索引操作
    grouped_data = df_sorted.groupby('group')

    for group, group_data in grouped_data:
        segment_times, segment_x_means, segment_y_means = compute_segment_data_vectorized(group_data, max_segments)

        group_segment_data[group] = {
            'times': segment_times,
            'x_means': segment_x_means,
            'y_means': segment_y_means
        }

        # 收集全局范围数据
        all_x_values.extend(segment_x_means)
        all_y_values.extend(segment_y_means)

    # 创建图形和子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=False, sharey=True,
                             squeeze=False)

    # 绘制每个组的图表
    for i, group in enumerate(unique_groups):
        # 计算子图位置
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # 创建次坐标轴
        ax2 = ax.twinx()
        if all_x_values:  # 避免空列表错误
            ax2.set_ylim(min(all_x_values), max(all_x_values))

        # 获取预计算的分段数据
        segment_data = group_segment_data[group]
        segment_times = segment_data['times']
        segment_x_means = segment_data['x_means']
        segment_y_means = segment_data['y_means']

        # 绘制图表
        if len(segment_times) > 0:
            ax.plot(segment_times, segment_y_means, linewidth=2, marker="o", markersize=4, color='blue', label=y_label)
            ax2.plot(segment_times, segment_x_means, linewidth=2, marker="o", markersize=4, color='red', label=x_label)

        # 设置子图标题和标签
        ax.set_title(f"{group}", fontsize=15)
        ax.set_xlabel(time_label, fontsize=15)
        ax.set_ylabel(y_label, fontsize=15)
        ax2.set_ylabel(x_label, fontsize=15)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center',bbox_to_anchor=(0.5, -0.10),fontsize=12)

    fig.autofmt_xdate(rotation=45)

    # 隐藏空的子图
    for i in range(n_groups, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    # 添加总标题
    if title:
        fig.suptitle(title, fontsize=18, y=0.95)
    plt.tight_layout()

    return fig, axes


def plot_rf_corr(df: pd.DataFrame, y_col: str, top_features: list, n: int = 10, save_path: str = None, title: str = None):
    """
    绘制因变量和前n个重要特征的相关性热图
    :param df: 数据集DataFrame
    :param y_col: 因变量列名
    :param top_features: 重要特征名列表
    :param n: 取前n个特征
    :param save_path: 图片保存路径
    :param title: 图标题
    :return: corr_matrix
    """
    targets = [y_col] + top_features[:n]
    return plot_corr(df, targets, save_path=save_path, title=title)


def plot_y_histogram(y_series, bins=20):
    """
    绘制Y列直方图

    Args:
        y_series: pandas Series，Y列数据
        bins: 直方图bin数量，默认20

    Returns:
        matplotlib.figure.Figure: 直方图fig对象，失败时返回None
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(y_series, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')

        # 使用Series的名称作为标题和x轴标签
        series_name = y_series.name if y_series.name is not None else 'Y'
        ax.set_title(f'Distribution of {series_name}', fontsize=14)
        ax.set_xlabel(series_name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)

        # 计算统计量
        mean_val = y_series.mean()
        std_val = y_series.std()
        min_val = y_series.min()
        max_val = y_series.max()

        # 根据数值大小自动调整显示格式
        def format_number(num):
            """根据数值大小自动选择显示格式"""
            if abs(num) < 1e-3 or abs(num) > 1e6:
                # 对于很小的数或很大的数使用科学计数法
                return f"{num:.2e}"
            elif abs(num) < 1:
                # 对于小于1的数，保留更多小数位
                return f"{num:.6f}".rstrip('0').rstrip('.')
            else:
                # 对于正常范围的数，保留2位小数
                return f"{num:.2f}"

        # 添加统计量：均值 标准差 范围[最小值-最大值]
        ax.text(0.02, 0.98, f'mean: {format_number(mean_val)}', transform=ax.transAxes, va='top')
        ax.text(0.02, 0.93, f'std: {format_number(std_val)}', transform=ax.transAxes, va='top')
        ax.text(0.02, 0.88, f'range: {format_number(min_val)} - {format_number(max_val)}', transform=ax.transAxes, va='top')

        # 如果数值很小，调整x轴刻度格式
        if abs(mean_val) < 1e-3 or abs(mean_val) > 1e6:
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

        return fig
    except Exception as e:
        print(f"绘制Y列直方图时出错: {str(e)}")
        return None
