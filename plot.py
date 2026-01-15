"""
functions of plot figures
"""
import warnings
from typing import Optional

from matplotlib.figure import Figure
from matplotlib.pylab import mpl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import QhullError
from typing import Tuple
from eval import compute_segment_data_qcut

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

    corr_matrix = dataset[targets].corr().fillna(0)
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



def generate_shap_figures(model, X, fig_path=None, n_features=5):
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
    import shap
    n_features = min(X.shape[1], n_features)

    # Determine the appropriate SHAP explainer
    if type(model).__name__ in ["GradientBoostingRegressor", "RandomForestRegressor", "XGBRegressor", "LightGBMRegressor"]:
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
        # 构建依赖图的文件路径
        base_path = fig_path.replace('.png', '')
        dependent_fig_path = f"{base_path}_{feature_name}.png"
        generate_shap_dependent_plot(feature_name, shap_values, X, save_path=dependent_fig_path)
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

    try:
        # use linear interpolation by default
        Z = griddata((x, y), z, (X, Y), method='linear')
    except QhullError:  # when initial simplex is not convex
        # use nearest neighbor interpolation
        Z = griddata((x, y), z, (X, Y), method='nearest')

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


def plot_group_scatter(x: pd.Series, y: pd.Series, groups: Optional[pd.Series] = None, figsize=(15, 10),
                       title=None, return_data=False, index=None):
    """
    对x和y进行分组，并绘制，每个分组的x y 绘制一个散点图，添加线性趋势线，并显示公式和R^2值

    Args:
        x: pd.Series - x axis data
        y: pd.Series - y axis data
        groups: pd.Series, optional - group data, if None then default to one group
        figsize: tuple - figure size
        title: str, optional - figure title
        return_data: bool, optional - whether to return data, default is False
        index: pd.Series, optional - index data, if None then use index of x
    """
    # 参数验证
    if len(x) != len(y):
        raise ValueError("x列和y列的长度必须相同")

    # label
    x_label = x.name if isinstance(x, pd.Series) and x.name is not None else 'X轴'
    y_label = y.name if isinstance(y, pd.Series) and y.name is not None else 'Y轴'

    # 创建DataFrame并删除缺失值
    df = pd.DataFrame({'x': x, 'y': y})
    if index is not None:
        df['index'] = index
    else:
        df['index'] = x.index
    if groups is not None:
        df['group'] = groups
    df = df.dropna()
    if len(df) == 0:
        raise ValueError("dataset has no valid data")

        # 如果groups为None，创建默认分组
    if 'group' not in df.columns:
        df['group'] = ' '

    # 获取唯一的分组
    unique_groups = df['group'].unique()

    # exclude groups with less than 2 points (at least 2 points are needed to fit a line)
    unique_groups = [group for group in unique_groups if len(df[df['group'] == group]) >= 2]
    n_groups = len(unique_groups)  # number of valid groups

    if n_groups == 0:
        raise ValueError("no enough data in groups (at least 2 points are needed for each group)")

    # calculate the number of rows and columns, make the subplot layout as close to a square as possible
    n_cols = int(np.ceil(np.sqrt(n_groups)))
    n_rows = int(np.ceil(n_groups / n_cols))

    # 创建图形和子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=False, sharey=False,
                             squeeze=False)

    # 用于存储每个组的回归统计信息（当return_data=True时使用）
    group_stats = {}

    # 绘制每个组的图表
    for i, group in enumerate(unique_groups):
        # 计算子图位置
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # 获取当前组的数据
        group_data = df[df['group'] == group]
        group_x = group_data['x'].values
        group_y = group_data['y'].values

        # 绘制散点图
        ax.scatter(group_x, group_y, alpha=0.4, color="red", s=30, linewidth=0.5)

        # 线性回归拟合
        slope = None
        intercept = None
        r2 = None

        if len(group_x) >= 2 and min(group_x) != max(group_x) and max(group_y) != min(group_y):
            # 使用numpy.polyfit进行线性拟合（1次多项式）
            coeffs = np.polyfit(group_x, group_y, 1)
            slope = coeffs[0]  # 斜率
            intercept = coeffs[1]  # 截距

            # 计算R²值
            y_pred = np.polyval(coeffs, group_x)
            ss_res = np.sum((group_y - y_pred) ** 2)
            ss_tot = np.sum((group_y - np.mean(group_y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # plot the trend line
            x_line = np.linspace(group_x.min(), group_x.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(x_line, y_line, 'b-', linewidth=2, label='trend line')

            # display the formula and R² value on the figure
            if intercept >= 0:
                formula_text = f'$Y = {slope:.3g}x + {intercept:.3g}$'
            else:
                formula_text = f'$Y = {slope:.3g}x - {abs(intercept):.3g}$'

            # display the formula and R² value on the top right corner of the figure
            text_str = f'{formula_text}\n$R^2={r2:.3g}$'
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

        # save the statistics information for return_data
        group_stats[group] = {
            'slope': slope,
            'intercept': intercept,
            'r2': r2
        }

        # set the title and labels of the subplot
        ax.set_title(f"{group}", fontsize=15)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', fontsize=9)

    # 隐藏空的子图
    for i in range(n_groups, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    # add title
    if title:
        fig.suptitle(title, fontsize=18, y=0.95)
    plt.tight_layout()
    if return_data:
        fig_data = []
        for group in unique_groups:
            group_data = df[df['group'] == group]
            stats = group_stats.get(group, {})
            fig_data.append({
                'index': group_data['index'].tolist(),
                'x': group_data['x'].tolist(),
                'x_max': float(group_data['x'].max()),
                'x_min': float(group_data['x'].min()),
                'y': group_data['y'].tolist(),
                'y_max': float(group_data['y'].max()),
                'y_min': float(group_data['y'].min()),
                'title': str(group),
                'slope': stats.get('slope'),
                'intercept': stats.get('intercept'),
                'r2': stats.get('r2')
            })
        return fig_data
    else:
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
                            figsize=(15, 10), title=None, max_segments: int = 30, return_data: bool = False):
    """
    对时间序列数据进行分组，并按时间顺序分段取平均值后绘制，提高大数据量时的绘图性能

    Args:
        time_col: pd.Series - 时间列数据，用于排序和分段
        x: pd.Series - X轴数据，将显示在次坐标轴上
        y: pd.Series - Y轴数据，将显示在主坐标轴上
        groups: pd.Series, optional - 分组数据，如果为None则默认分为一组
        figsize: tuple - 图形尺寸
        title: str, optional - 图形标题
        min_periods: int - 滚动窗口计算所需的最小观测数，默认为1
        max_segments: int - 每个分组最多分成的段数，默认为30
        return_data: bool, optional - 是否返回数据，默认为False
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
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
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

    # 一次性计算所有组的分段数据，避免重复计算
    group_segment_data = {}
    all_x_values = []
    all_y_values = []

    # 预先排序所有数据，避免在循环中重复排序
    df_sorted = df.sort_values(by=['group', 'time'])

    # 使用groupby进行高效分组，避免重复的布尔索引操作
    grouped_data = df_sorted.groupby('group')

    for group, group_data in grouped_data:
        segment_times, segment_x_means, segment_y_means = compute_segment_data_qcut(group_data, max_segments)

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
        segment_times = [str(i) for i in pd.to_datetime(segment_data['times'], unit='s')]
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

        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.10), fontsize=12)

        # 为每个子图单独设置x轴标签格式和旋转
        if len(segment_times) > 0:
            # 根据数据点数量调整x轴标签显示
            if len(segment_times) > 10:
                # 如果数据点较多，只显示部分标签
                step = max(1, len(segment_times) // 10)
                ax.set_xticks(range(0, len(segment_times), step))
                ax.set_xticklabels([segment_times[i] for i in range(0, len(segment_times), step)], rotation=45, ha='right')
            else:
                # 如果数据点较少，显示所有标签
                ax.set_xticks(range(len(segment_times)))
                ax.set_xticklabels(segment_times, rotation=45, ha='right')

    # 隐藏空的子图
    for i in range(n_groups, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    # 添加总标题
    if title:
        fig.suptitle(title, fontsize=18, y=0.95)
    plt.tight_layout()

    if return_data:
        # 返回绘图数据
        fig_data = []
        for group in unique_groups:
            segment_data = group_segment_data.get(group, {})
            segment_times = segment_data.get('times', np.array([]))
            segment_x_means = segment_data.get('x_means', np.array([]))
            segment_y_means = segment_data.get('y_means', np.array([]))

            # 转换numpy数组为列表，处理时间类型
            times_list = segment_times.tolist() if len(segment_times) > 0 else []
            # 如果时间不是字符串，转换为字符串格式
            if len(times_list) > 0 and not isinstance(times_list[0], str):
                try:
                    times_list = [pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(t) else None for t in times_list]
                except Exception:
                    times_list = [str(t) if pd.notna(t) else None for t in times_list]

            fig_data.append({
                'title': str(group),
                'time': times_list,
                'x': segment_x_means.tolist() if len(segment_x_means) > 0 else [],
                'x_max': float(segment_x_means.max()),
                'x_min': float(segment_x_means.min()),
                'y': segment_y_means.tolist() if len(segment_y_means) > 0 else [],
                'y_max': float(segment_y_means.max()),
                'y_min': float(segment_y_means.min()),
                'x_label': x_label,
                'y_label': y_label,
                'time_label': time_label
            })
        return fig_data
    else:
        return fig, axes


def plot_rf_corr(df: pd.DataFrame, y_col: str, top_features: list, n: int = 10, save_path: str = None,
                 title: str = None):
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
        y_series: pandas Series, Y column data
        bins: number of bins for the histogram, default is 20

    Returns:
        matplotlib.figure.Figure
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
        ax.text(0.02, 0.88, f'range: {format_number(min_val)} - {format_number(max_val)}', transform=ax.transAxes,
                va='top')

        # 如果数值很小，调整x轴刻度格式
        if abs(mean_val) < 1e-3 or abs(mean_val) > 1e6:
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

        return fig
    except Exception as e:
        print(f"绘制Y列直方图时出错: {str(e)}")
        return None

def generate_shap_dependent_plot(feature_name, shap_values, x, save_path=None):
    """
    generate shap dependent plot
    :param feature_name: feature name
    :param shap_values: shap values
    :param x: input data
    :param save_path: figure save path
    :return: figure
    """
    import shap
    shap.dependence_plot(feature_name, shap_values, x, show=False, interaction_index=None)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
