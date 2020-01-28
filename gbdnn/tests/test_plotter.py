import unittest
import json
import os

import numpy as np
from gbdnn.plotter import Plotter

dir_test_data = os.path.join(os.path.dirname(__file__), 'test.json')

class PlotterTest(unittest.TestCase):

    def setUp(self):
        self.plotter1 = Plotter()
        self.plotter2 = Plotter(span=0.4, robust=False)
        with open(dir_test_data) as f:
            compositions, temperatures, adsorptions = json.load(f)
        self.x = np.stack((compositions, temperatures), axis=1)
        self.y = np.array(adsorptions)[:, np.newaxis]

    def test_smooth(self):
        n = self.x.shape[0]
        self.plotter1.smooth(self.x, self.y)
        self.assertFalse(np.all(self.plotter1.model.robust_weights == np.ones((n, 1))))

        self.plotter2.smooth(self.x, self.y, iteration=10)
        self.assertTrue(np.all(self.plotter2.model.robust_weights == np.ones((n, 1))))

    def test_get_cd_data(self):
        self.plotter1.smooth(self.x, self.y)
        x, y, z = self.plotter1.get_cd_data(num=50)
        self.assertListEqual([x.shape[0], x.shape[0]], [50, 50])
        self.assertListEqual([y.shape[0], y.shape[0]], [50, 50])
        self.assertListEqual([z.shape[0], z.shape[0]], [50, 50])

        self.plotter2.smooth(self.x, self.y, iteration=30)
        x, y, z = self.plotter2.get_cd_data(num=100)
        self.assertListEqual([x.shape[0], x.shape[0]], [100, 100])
        self.assertListEqual([y.shape[0], y.shape[0]], [100, 100])
        self.assertListEqual([z.shape[0], z.shape[0]], [100, 100])

    def test_get_plot(self):
        self.plotter1.smooth(self.x, self.y)
        ax1 = self.plotter1.get_plot(formatter='surface')
        ax2 = self.plotter1.get_plot(formatter='flat')

        self.plotter2.smooth(self.x, self.y, iteration=30)
        ax3 = self.plotter2.get_plot(formatter='surface')
        ax4 = self.plotter2.get_plot(formatter='flat', vmin=0, vmax=25)

        self.assertRaises(ValueError, self.plotter2.get_plot, 'null')

if __name__ == '__main__':
    unittest.main()
