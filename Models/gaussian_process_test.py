"""
An implementation of GPs from scratch; used to understand how GPs work only.
Original code is from:
 http://krasserm.github.io/2020/11/04/gaussian-processes-classification/
https://github.com/krasserm/bayesian-machine-learning/blob/dev/gaussian-processes/gaussian_processes_util.py
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import bernoulli
from scipy.special import expit as sigmoid

from sklearn.datasets import make_moons
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D



def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()


def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)



def plot_data_1D(X, t):
    class_0 = t == 0
    class_1 = t == 1

    plt.scatter(X[class_1], t[class_1], label='Class 1', marker='x', color='red')
    plt.scatter(X[class_0], t[class_0], label='Class 0', marker='o', edgecolors='blue', facecolors='none')


def plot_data_2D(X, t):
    class_1 = np.ravel(t == 1)
    class_0 = np.ravel(t == 0)

    plt.scatter(X[class_1, 0], X[class_1, 1], label='Class 1', marker='x', c='red')
    plt.scatter(X[class_0, 0], X[class_0, 1], label='Class 0', marker='o', edgecolors='blue', facecolors='none')

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')


def plot_pt_2D(grid_x, grid_y, grid_z):
    plt.contourf(grid_x, grid_y, grid_z, cmap='plasma', alpha=0.3, levels=np.linspace(0, 1, 11))
    plt.colorbar(format='%.2f')


def plot_db_2D(grid_x, grid_y, grid_z, decision_boundary=0.5):
    levels = [decision_boundary]
    cs = plt.contour(grid_x, grid_y, grid_z, levels=levels, colors='black', linestyles='dashed', linewidths=2)
    plt.clabel(cs, fontsize=20)



def generate_animation(theta_steps, X_m_steps, X_test, f_true, X, y, sigma_y, phi_opt, q, interval=100):
    fig, ax = plt.subplots()

    line_func, = ax.plot(X_test, f_true, label='Latent function', c='k', lw=0.5)
    pnts_ind = ax.scatter([], [], label='Inducing variables', c='m')

    line_pred, = ax.plot([], [], label='Prediction', c='b')
    area_pred = ax.fill_between([], [], [], label='Epistemic uncertainty', color='r', alpha=0.1)

    ax.set_title('Optimization of a sparse Gaussian process')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-3, 3.5)
    ax.legend(loc='upper right')

    def plot_step(i):
        theta = theta_steps[i]
        X_m = X_m_steps[i]

        mu_m, A_m, K_mm_inv = phi_opt(theta, X_m, X, y, sigma_y)
        f_test, f_test_cov = q(X_test, theta, X_m, mu_m, A_m, K_mm_inv)
        f_test_var = np.diag(f_test_cov)
        f_test_std = np.sqrt(f_test_var)

        ax.collections.clear()
        pnts_ind = ax.scatter(X_m, mu_m, c='m')

        line_pred.set_data(X_test, f_test.ravel())
        area_pred = ax.fill_between(X_test.ravel(),
                                    f_test.ravel() + 2 * f_test_std,
                                    f_test.ravel() - 2 * f_test_std,
                                    color='r', alpha=0.1)

        return line_func, pnts_ind, line_pred, area_pred

    result = animation.FuncAnimation(fig, plot_step, frames=len(theta_steps), interval=interval)

    # Prevent output of last frame as additional plot
    plt.close()

    return result


np.random.seed(0)

X = np.arange(0, 5, 0.2).reshape(-1, 1)
X_test = np.arange(-2, 7, 0.1).reshape(-1, 1)

a = np.sin(X * np.pi * 0.5) * 2
t = bernoulli.rvs(sigmoid(a))

plot_data_1D(X, t)
plt.title('1D training dataset')
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.yticks([0, 1])
plt.legend()
plt.show()


def kernel(X1, X2, theta):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
        theta: Kernel parameters

    Returns:
        (m x n) matrix
    """

    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return theta[1] ** 2 * np.exp(-0.5 / theta[0] ** 2 * sqdist)

def K_(X, theta, diag_only=False, nu=1e-5):
    """Helper to apply kernel function."""
    if diag_only:
        # Specific solution for isotropic
        # squared exponential kernel.
        return theta[1] ** 2 + nu
    else:
        return kernel(X, X, theta) + nu * np.eye(X.shape[0])

def W_(a):
    """Helper to compute matrix W."""
    r = sigmoid(a) * (1 - sigmoid(a))
    return np.diag(r.ravel())


def posterior_mode(X, t, K_a, max_iter=10, tol=1e-9):
    """
    Computes the mode of posterior p(a|t).
    """
    a_h = np.zeros_like(t)
    I = np.eye(X.shape[0])

    for i in range(max_iter):
        W = W_(a_h)
        Q_inv = np.linalg.inv(I + W @ K_a)
        a_h_new = (K_a @ Q_inv).dot(t - sigmoid(a_h) + W.dot(a_h))
        a_h_diff = np.abs(a_h_new - a_h)
        a_h = a_h_new

        if not np.any(a_h_diff > tol):
            break

    return a_h

def nll_fn(X, t):
    """
    Returns the negative log-likelihood function for data X, t.
    """

    t = t.ravel()

    def nll(theta):
        K_a = K_(X, theta)
        K_a_inv = np.linalg.inv(K_a)

        # posterior mode depends on theta (via K)
        a_h = posterior_mode(X, t, K_a).ravel()
        W = W_(a_h)

        ll = - 0.5 * a_h.T.dot(K_a_inv).dot(a_h) \
             - 0.5 * np.linalg.slogdet(K_a)[1] \
             - 0.5 * np.linalg.slogdet(W + K_a_inv)[1] \
             + t.dot(a_h) - np.sum(np.log(1.0 + np.exp(a_h)))

        return -ll

    return nll

def predict_a(X_test, X, t, theta):
    """
    Computes the mean and variance of logits at points X_test
    given training data X, t and kernel parameters theta.
    """
    K_a = K_(X, theta)
    K_s = kernel(X, X_test, theta)
    a_h = posterior_mode(X, t, K_a)

    W_inv = np.linalg.inv(W_(a_h))
    R_inv = np.linalg.inv(W_inv + K_a)

    a_test_mu = K_s.T.dot(t - sigmoid(a_h))
    # Compute variances only (= diagonal) instead of full covariance matrix
    a_test_var = K_(X_test, theta, diag_only=True) - np.sum((R_inv @ K_s) * K_s, axis=0).reshape(-1, 1)

    return a_test_mu, a_test_var


def predict_pt(X_test, X, t, theta):
    """
    Computes the probability of t=1 at points X_test
    given training data X, t and kernel parameters theta.
    """
    a_mu, a_var = predict_a(X_test, X, t, theta)
    kappa = 1.0 / np.sqrt(1.0 + np.pi * a_var / 8)
    return sigmoid(kappa * a_mu)



res = minimize(nll_fn(X, t), [1, 1],
               bounds=((1e-3, None), (1e-3, None)),
               method='L-BFGS-B')

theta = res.x

print(f'Optimized theta = [{theta[0]:.3f}, {theta[1]:.3f}], negative log likelihood = {res.fun:.3f}')


pt_test = predict_pt(X_test, X, t, theta)

plot_data_1D(X, t)
plt.plot(X_test, pt_test, label='Prediction', color='green')
plt.axhline(0.5, X_test.min(), X_test.max(), color='black', ls='--', lw=0.5)
plt.title('Predicted class 1 probabilities')
plt.xlabel('$x$')
plt.ylabel('$p(t_*=1|\mathbf{t})$')
plt.legend()
plt.show()


a_test_mu, a_test_var = predict_a(X_test, X, t, theta)

a_test_mu = a_test_mu.ravel()
a_test_var = a_test_var.ravel()

plt.plot(X_test, a_test_mu, label='logits mean $\mu_{a_*}$', color='green', alpha=0.3)
plt.fill_between(X_test.ravel(),
                 a_test_mu + a_test_var,
                 a_test_mu - a_test_var,
                 label='logits variance $\sigma^2_{a_*}$',
                 color='lightcyan')
plt.xlabel('$x$')
plt.legend()
plt.show()