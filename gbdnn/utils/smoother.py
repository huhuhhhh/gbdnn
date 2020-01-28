"""
Regression algorithm for smoothing the rigidity of grain boundary
complexion diagram.
"""

import numpy as np
from scipy.spatial.distance import cdist


class LocalRegressor:
    """
    Implementation of local regression algorithm.
    """
    def __init__(self, span=0.25, robust=True):
        """
        Args:
            span (float): The fraction of subset of data used in each fitting.
                Default to 0.25.
            robust (bool): Whether to apply robust weights preventing the
                regression function from potential outliers. Default to True.
        """
        if span < 0 or span > 1:
            raise ValueError("Value of span should be between 0 and 1!")

        self.span = span
        self.robust = robust
        self.x = None
        self.y = None
        self.extended_x = None
        self.robust_weights = None
        self.fitted = False

    def fit(self, x, y, iteration=20):
        """
        Args:
            x (numpy.array): Array with dim (n, d), where n is the number of data
                and d is the features dimension.
            y (numpy.array): Target array with dim (n, )
            iteration (int): The number of robustifying iterations to perform local
                regression. Default to 20.
        """
        self.fitted = False
        if len(x.shape) == 1:
            x = x[:, np.newaxis]

        self.x = x
        self.y = y

        n = x.shape[0]
        extended_x = np.concatenate((np.ones((n, 1)), x), axis=1)
        self.extended_x = extended_x

        num_neighbors = int(np.ceil(n * self.span))
        self.num_neighbors = num_neighbors
        max_dists = [np.sort(np.linalg.norm(x[i, :] - x, axis=1))[num_neighbors]
                     for i in range(n)]

        local_weights = np.clip(cdist(x, x, 'euclidean') / max_dists, 0.0, 1.0)
        local_weights = (1 - local_weights ** 3) ** 3

        robust_weights = np.ones((n, 1))

        for _ in range(iteration):
            y_pred = self.lowess(x, local_weights, robust_weights)
            if self.robust:
                robust_weights = self.residual(y_pred)

        self.robust_weights = robust_weights
        self.fitted = True

    def predict(self, sample_x):
        """
        Args:
            sample_x (numpy.array): Samples x to predict.

        Returns:
            y_pred (numpy.array): Array with dim (n, 1), where each element is the
                predicted target of samples_x.
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted!")
        if len(sample_x.shape) == 1:
            sample_x = sample_x[:, np.newaxis]

        if sample_x.shape[1] != self.x.shape[1]:
            raise ValueError("Dimension of inputs is not matched!")

        n = sample_x.shape[0]
        max_dists = [np.sort(np.linalg.norm(sample_x[i, :] - self.x, axis=1))[self.num_neighbors]
                     for i in range(n)]
        local_weights = np.clip(cdist(self.x, sample_x, 'euclidean') / max_dists, 0.0, 1.0)
        local_weights = (1 - local_weights ** 3) ** 3

        y_pred = self.lowess(sample_x, local_weights, self.robust_weights)

        return y_pred

    def lowess(self, x, local_weights, robust_weights):
        """
        Args:
            x (numpy.array): Array with dim (m, d), where m is the number of data
                and d is the features dimension.
            local_weights (numpy.array): Array with dim (n, m), where each element
                represents locally distance-dependent weights between each pair of
                sample_x and x fitted. In the fit method, sample_x is the same as x
                fitted.
            robust_weights (numpy.array): Robust weights to prevent locally regression
                from potential outliers.

        Returns:
            y_pred (numpy.array): Array with dim (m, 1), where each element is the
                predicted target of samples x.
        """
        n = x.shape[0]
        y_pred = np.zeros((n, 1))
        extended_x = np.concatenate((np.ones((n, 1)), x), axis=1)

        for i in range(n):
            weights = np.sqrt(robust_weights * local_weights[:, [i]])

            w_x = weights * self.extended_x
            w_y = weights * self.y

            # least square fitting
            theta = np.dot(np.linalg.inv(np.dot(w_x.T, w_x)), np.dot(w_x.T, w_y))
            y_pred[i] = np.dot(extended_x[i, :], theta)

        return y_pred

    def residual(self, y_pred):
        """
        Args:
            y_pred (numpy.array): Array with dim (n, 1), where each element is the
                predicted target of samples x.

        Returns:
            robust_weights (numpy.array): Robust weights.
        """
        residuals = self.y - y_pred
        median = np.median(np.abs(residuals.ravel()))

        robust_weights = np.clip(residuals / (6.0 * median), -1, 1)
        robust_weights = (1 - robust_weights ** 2) ** 2

        return robust_weights
