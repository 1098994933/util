import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def draw_corr(dataset, targets, path="corr.png"):
    """
    draw corr heatmap
    :param dataset:
    :param targets:
    :param path:
    :return:
    """
    # Target corr
    corr = dataset[targets].corr()
    print(corr)
    with sns.axes_style("white"):
        ax = sns.heatmap(corr, annot=True, annot_kws={"size": 10}, fmt=".2f",
                         cmap="rainbow", linewidths=0, vmin=-1, vmax=1, square=True,

                         )
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.savefig(f'./figures/{path}', bbox_inches='tight')