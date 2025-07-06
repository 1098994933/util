import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


def fit_2d_gaussian(data):
    """
    fit a gaussian by 2d-array

    param:
        data: 二维数据点，形状为 (N, 2) 的NumPy数组。

    return:
        mean: 估计得到的高斯分布均值，形状为 (2,) 的NumPy数组。
        cov: 估计得到的高斯分布协方差矩阵，形状为 (2, 2) 的NumPy数组。
        rho: 估计得到的高斯分布相关系数。
    """

    # opt function for MLE
    def negative_log_likelihood(params):
        mu, std, rho = params[:2], params[2:4], params[4]
        cov = np.array([[std[0] ** 2, std[0] * std[1] * rho], [std[0] * std[1] * rho, std[1] ** 2]])
        mvn = multivariate_normal(mu, cov)
        nll = -np.mean(np.log(mvn.pdf(data)))
        return nll

    # minimize the negative_log_likelihood
    init_params = np.concatenate((np.mean(data, axis=0), np.std(data, axis=0), [0]))
    result = minimize(negative_log_likelihood, init_params, method='BFGS')
    fitted_params = result.x

    mean, std, rho = fitted_params[:2], fitted_params[2:4], fitted_params[4]
    cov = np.array([[std[0] ** 2, std[0] * std[1] * rho], [std[0] * std[1] * rho, std[1] ** 2]])

    return mean, cov, rho


def probability_in_rectangle_monte_carlo(mu, cov, rectangle, n_samples=30000):
    """
    estimate probability of the sample in a rectangle scope by monte carlo
    rectangle: [(x_min, x_max), (y_min, y_max))
    return: prob
    """
    # create normal distribution
    distribution = multivariate_normal(mean=mu, cov=cov)

    # sampling from distribution
    samples = distribution.rvs(size=n_samples)

    # count the number in rectangle x_min< x < x_max and y_min< y < y_max
    inside_rectangle = np.sum(np.all([rectangle[0][0] <= samples[:, 0], samples[:, 0] < rectangle[1][0],
                                      rectangle[0][1] <= samples[:, 1], samples[:, 1] < rectangle[1][1]], axis=0))

    probability = inside_rectangle / n_samples
    return probability
