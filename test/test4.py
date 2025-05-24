import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import savgol_filter  # 导入Savitzky-Golay滤波器

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def generate_year_data(start_date, output_file=None):
    """
    生成一年的数据，其中1-10月方差较小，11-12月方差较大

    参数:
    start_date (str): 开始日期，格式为 'YYYY-MM-DD'
    output_file (str, optional): 输出CSV文件的路径，如果为None则不保存

    返回:
    pandas.DataFrame: 包含生成数据的DataFrame
    """
    # 将字符串转换为日期
    start = datetime.strptime(start_date, '%Y-%m-%d')
    # 生成一年的日期列表（365天）
    dates = [start + timedelta(days=i) for i in range(365)]

    # 生成数据 - 基础值为一个缓慢增长的趋势
    base_values = np.linspace(100, 101, 365)

    # 为不同月份设置不同的方差
    # 1-10月方差较小，11-12月方差较大（两倍）
    small_std = 5  # 1-10月的标准差
    large_std = 10  # 11-12月的标准差

    # 为每一天生成随机噪声
    noise = np.zeros(365)
    for i, date in enumerate(dates):
        if date.month <= 10:
            noise[i] = np.random.normal(0, small_std)
        else:
            noise[i] = np.random.normal(0, large_std)

    # 计算最终数据值
    values = base_values + noise

    # 创建DataFrame
    df = pd.DataFrame({
        '日期': dates,
        '数值': values
    })

    # 如果指定了输出文件，则保存为CSV
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"数据已保存到 {output_file}")

    return df


def smooth_data(y, window_length=7, polyorder=3):
    """
    使用Savitzky-Golay滤波器平滑数据

    参数:
    y (array-like): 需要平滑的数据
    window_length (int): 滤波器窗口长度，必须是奇数
    polyorder (int): 多项式阶数

    返回:
    array-like: 平滑后的数据
    """
    return savgol_filter(y, window_length, polyorder)


def plot_data(df, title=""):
    """
    绘制数据的折线图

    参数:
    df (pandas.DataFrame): 包含数据的DataFrame
    title (str, optional): 图表标题
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 在主坐标轴上绘制原始数据
    ax1.plot(df['日期'], df['数值'], label='Y数据值', linewidth=1.5, color='#3498db', alpha=0.7)
    ax1.set_ylabel('电性参数', fontsize=12, color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')

    # 创建次坐标轴
    ax2 = ax1.twinx()

    # 在次坐标轴上绘制平滑曲线
    smoothed_values = smooth_data(df['数值'])
    df['平滑数值'] = smoothed_values
    ax2.plot(df['日期'], smoothed_values, label='关键自变量', linewidth=2.5, color='#e74c3c')
    ax2.set_ylabel('关键自变量的值', fontsize=12, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    # 标记11月和12月（方差较大的部分）
    nov_start = df[df['日期'].dt.month == 11].index.min()
    dec_end = df[df['日期'].dt.month == 12].index.max()

    ax1.axvspan(df.loc[nov_start, '日期'], df.loc[dec_end, '日期'],
                color='#e74c3c', alpha=0.1, label='高方差区域')

    # 添加标题和标签
    plt.title(title, fontsize=16)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 合并两个坐标轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    # 设置x轴日期格式
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 生成数据（从2024年1月1日开始）
    data = generate_year_data('2024-01-01', output_file='year_data.csv')

    # 绘制数据可视化图
    plot_data(data, "")

    # 打印统计信息
    early_months = data[data['日期'].dt.month <= 10]
    late_months = data[data['日期'].dt.month > 10]

    print("\n数据统计信息:")
    print(f"1-10月方差: {early_months['数值'].var():.2f}")
    print(f"11-12月方差: {late_months['数值'].var():.2f}")
    print(f"方差比例(11-12月/1-10月): {late_months['数值'].var() / early_months['数值'].var():.2f}")