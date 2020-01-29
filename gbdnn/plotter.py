import os
import numpy as np
from monty.serialization import loadfn
from gbdnn.utils import smoother

module_dir = os.path.dirname(__file__)
compositions, temperatures \
    = loadfn(os.path.join(module_dir, "utils", "params", "solidus_line.json"))
color_list = loadfn(os.path.join(module_dir, "utils", "params", "parula.json"))


class Plotter:
    """
    A plotter class for plotting grain boundary complexion diagram.
    """
    def __init__(self, regressor='LocalRegressor', **kwargs):
        """
        Args:
            regressor (str): Regressor type. Currently only LocalRegressor is
                available.
            kwargs: kwargs to be passed to regressor.
        """
        lr = getattr(smoother, regressor)
        self.model = lr(**kwargs)

    def smooth(self, x, y, **kwargs):
        """
        Args:
            x (numpy.array): Array with dim (n, d), where n is the number of
                data and d is the feature dimension.
            y (numpy.array): Target array with dim (n,).
            kwargs: kwargs to be passed to model.fit
        """
        self.model.fit(x, y, **kwargs)

    def get_cd_data(self, num=200):
        """
        Get the uniformly sampled data for grain boundary complexion diagram.

        Args:
            num (int): Number of samples to generate. Default to 200.
                Larger value will result in figure of higher resolution.
        """
        x = np.linspace(0, 1, num)
        y = np.linspace(0.5, 1, num)

        x, y = np.meshgrid(x, y)
        reshaped_xy = np.concatenate((np.reshape(x, (num * num, 1)),
                                      np.reshape(y, (num * num, 1))), axis=1)
        z = np.reshape(self.model.predict(reshaped_xy), (num, num))

        return x, y, z

    def get_plot(self, formatter='flat', **kwargs):
        """
        Args:
            formatter (str): 'flat' or 'surface'. 'flat' returns two dimensional
                projected complexion diagram. 'surface' returns three dimensional
                complexion diagram.
        """
        if formatter.startswith('f'):
            plt = self._get_flat_plot(**kwargs)
        elif formatter.startswith('s'):
            plt = self._get_surface_plot(**kwargs)
        else:
            raise ValueError("Invalid formatter to return figure!")
        return plt

    def show(self, *args, **kwargs):
        """
        Draw the complexion diagram using Matplotlib and show it.

        Args:
            args: Passed to get_plot.
            kwargs: Passed to get_plot.
        """
        self.get_plot(*args, **kwargs).show()

    def _get_flat_plot(self, cmap=None, num=200, **kwargs):
        """
        Args:
            cmap (str): A colormap for the surface patches.
            num (int): Number of samples to generate. Default is 200.
                Larger value will result in figure of higher resolution.
            kwargs: kwargs to be passed into imshow method.
        """
        bound_x = np.array(compositions) / max(compositions)
        bound_y = np.array(temperatures) / max(temperatures)

        import matplotlib.pyplot as plt
        # plt.switch_backend('agg')

        if cmap is None:
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list('parula',
                                    color_list, len(color_list))

        fig, ax = plt.subplots(figsize=(12, 10))
        x, y, z = self.get_cd_data(num=num)

        ax.set_ylim(0.5, 1)
        ax.imshow(z, extent=(0, 1, 1, 0.5), cmap=cmap, aspect='auto', **kwargs)

        for xticklabel, yticklabel in zip(ax.xaxis.get_majorticklabels(),
                                          ax.yaxis.get_majorticklabels()):
            xticklabel.set_fontsize(24)
            yticklabel.set_fontsize(24)

        ax.set_xlabel('$X_{Bulk}\ /\ X_{Max\ Solubility}$', fontsize=30)
        ax.set_ylabel('$T\ /\ T_{m}$', fontsize=30)

        ax.plot(bound_x, bound_y, color='k', label='solidus line')
        ax.fill_between(np.concatenate((bound_x, np.ones_like(bound_x))),
                        np.concatenate((bound_y, bound_y[::-1])),
                        facecolors=[1, 1, 1])
        ax.legend(fontsize=24)

        return plt

    def _get_surface_plot(self, cmap=None, num=200):
        """
        Args:
            cmap (str): A colormap for the surface patches.
            num (int): Number of samples to generate. Default is 200.
                Larger value will result in figure of higher resolution.
        """
        import matplotlib.pyplot as plt
        # plt.switch_backend('agg')
        import mpl_toolkits.mplot3d.axes3d as p3
        fig = plt.figure(figsize=(12, 8))
        ax = p3.Axes3D(fig)
        x, y, z = self.get_cd_data(num=num)

        if cmap is None:
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list('parula',
                                    color_list, len(color_list))

        surf = ax.plot_surface(y, x, z, cmap=cmap, rstride=1, cstride=1,
                               antialiased=False)
        ax.invert_xaxis()
        ax.grid('False')
        for xticklabel, yticklabel, zticklabel in zip(ax.xaxis.get_majorticklabels(),
                                                      ax.yaxis.get_majorticklabels(),
                                                      ax.zaxis.get_majorticklabels()):
            xticklabel.set_fontsize(24)
            yticklabel.set_fontsize(24)
            zticklabel.set_fontsize(24)

        ax.set_xlabel('$T\ /\ T_{\\rm m}$', fontsize=24, labelpad=24)
        ax.set_ylabel('$X\ /\ X_{\\rm Max}$', fontsize=24, labelpad=24)
        ax.tick_params('z', pad=10)

        return plt
