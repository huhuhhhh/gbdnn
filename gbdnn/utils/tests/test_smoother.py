import unittest
import json
import os

import numpy as np
from scipy.spatial.distance import cdist
from gbdnn.utils.smoother import LocalRegressor

dir_test_data = os.path.join(os.path.dirname(__file__), 'test.json')


class LocalRegressorTest(unittest.TestCase):

    def setUp(self):
        self.regressor1 = LocalRegressor()
        self.regressor2 = LocalRegressor(span=0.4)
        self.regressor3 = LocalRegressor(span=0.5, robust=False)
        with open(dir_test_data) as f:
            compositions, temperatures, adsorptions = json.load(f)
        self.x = np.stack((compositions, temperatures), axis=1)
        self.y = np.array(adsorptions)[:, np.newaxis]

    def test_fit(self):
        n = self.x.shape[0]
        self.regressor1.fit(self.x, self.y)
        self.assertEqual(self.regressor1.num_neighbors, int(np.ceil(0.25 * n)))
        self.assertEqual(self.regressor1.extended_x.shape[1], 3)
        self.assertFalse(np.all(self.regressor1.robust_weights == np.ones((n, 1))))

        self.regressor2.fit(self.x, self.y)
        self.assertEqual(self.regressor2.num_neighbors, int(np.ceil(0.4 * n)))
        self.assertEqual(self.regressor2.extended_x.shape[1], 3)
        self.assertFalse(np.all(self.regressor2.robust_weights == np.ones((n, 1))))

        self.regressor3.fit(self.x, self.y)
        self.assertEqual(self.regressor3.num_neighbors, int(np.ceil(0.5 * n)))
        self.assertEqual(self.regressor3.extended_x.shape[1], 3)
        self.assertTrue(np.all(self.regressor3.robust_weights == np.ones((n, 1))))

    def test_predict(self):
        n = self.x.shape[0]
        self.regressor1.fit(self.x, self.y)
        y_pred = self.regressor1.predict(self.x)
        self.assertEqual(y_pred.shape[0], n)

    def test_lowess(self):
        n = self.x.shape[0]
        self.regressor1.fit(self.x, self.y)

        num_neighs = int(np.ceil(0.25 * n))
        max_dists = [np.sort(np.linalg.norm(self.x[i, :] - self.x, axis=1))[num_neighs]
                     for i in range(n)]
        local_weights = np.clip(cdist(self.x, self.x, 'euclidean') / max_dists, 0.0, 1.0)
        self.assertTrue(np.all(local_weights <= 1))
        self.assertTrue(np.all(local_weights >= 0))
        local_weights = (1 - local_weights ** 3) ** 3
        self.assertTrue(np.all(local_weights <= 1))
        self.assertTrue(np.all(local_weights >= 0))
        robust_weights = np.random.rand(n, 1)

        y_pred = self.regressor1.lowess(self.x, local_weights, robust_weights)
        self.assertEqual(y_pred.shape[0], n)

    def test_residual(self):
        n = self.x.shape[0]
        self.regressor1.fit(self.x, self.y)

        num_neighs = int(np.ceil(0.25 * n))
        max_dists = [np.sort(np.linalg.norm(self.x[i, :] - self.x, axis=1))[num_neighs]
                     for i in range(n)]
        local_weights = np.clip(cdist(self.x, self.x, 'euclidean') / max_dists, 0.0, 1.0)
        local_weights = (1 - local_weights ** 3) ** 3
        init_robust_weights = np.random.rand(n, 1)

        y_pred = self.regressor1.lowess(self.x, local_weights, init_robust_weights)
        robust_weights = self.regressor1.residual(y_pred)
        self.assertFalse(np.all(init_robust_weights == robust_weights))


if __name__ == '__main__':
    unittest.main()
