import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


def pca_2d_projection(X, Y):
    """
    Perform PCA on dataset X and plot the first two principal components.
    The color of the points in the scatter plot is determined by the values in Y.

    Parameters:
    X (array-like): The input data for PCA.
    Y (array-like): The values used to color the points in the scatter plot.

    Returns:
    plt.figure: A matplotlib figure object with the scatter plot.
    """
    # Perform PCA and reduce the data to two dimensions

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a scatter plot of the two principal components
    fig = plt.figure(figsize=(8, 6), dpi=300)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='viridis', label="Train", alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    return fig, pca
