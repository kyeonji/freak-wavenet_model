from welford import Welford
import numpy as np
# from scipy.stats import multivariate_normal
from pathlib import Path
import common_cnn_n as com
from tqdm import tqdm

param = com.yaml_load()


def calc_x(x, x_hat):
    x = x.squeeze()
    x_hat = x_hat.squeeze()
    Amp_x = np.mean(x, axis=0)  # time average
    Amp_x_hat = np.mean(x_hat, axis=0)

    error = abs(Amp_x - Amp_x_hat)
    return error


def fit_prob(train_set, model):
    num_files = train_set.shape[0]
    prob = Welford()
    for idx in tqdm(range(num_files)):
        train_data = train_set[idx, :, :]
        train_data = train_data.reshape(1, train_data.shape[0], train_data.shape[1])
        reconst = model.predict(train_data)
        x = calc_x(train_data[:, 31:, :], reconst)
        prob.add(x)

    mean = prob.mean
    var = prob.var

    return mean, var


def get_prob(x, mu, sigma):
    # mu = mu.squeeze()
    # sigma = sigma.squeeze()
    # covariance = np.expand_dims(np.diag(sigma), axis=0)
    prob = multivariate_normal(x, d=x.shape[-1], mean=mu, covariance=sigma)
    return prob


def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) *
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))
