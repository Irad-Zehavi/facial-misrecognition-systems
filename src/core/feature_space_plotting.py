from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize

from src.core.utils import DEVICE
from src.data.core.smart_dataset import SmartDataset


def artificial_cluster(x, y, z, normed=False, count=200):
    lengths = np.random.rand(count, 1) * 1.5
    x = np.random.normal((x, y, z), 0.07, (count, 3))
    x *= lengths
    return normalize(x) if normed else x


class FeatureSpaceFigure(object):
    def __init__(self, lim=None):
        self.fig: plt.Figure = plt.figure()
        self.ax: Axes3D = self.fig.add_subplot(projection='3d', box_aspect=(1, 1, 1))
        if lim:
            self.ax.set_xlim(-lim, lim)
            self.ax.set_ylim(-lim, lim)
            self.ax.set_zlim(-lim, lim)

    def plot_sphere(self, equatorial_plane=False):
        phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)

        self.ax.plot_surface(x, y, z, rcount=20, ccount=20, alpha=.3, color='white')

        if equatorial_plane:
            self.ax.plot_surface(x, y, 0*z, rcount=20, ccount=20, alpha=.15, color='white')

    def _scatter(self, points, color, s):
        self.ax.scatter(*points.transpose(), s=s, color=color)

    def plot_cluster(self, points, color, plot_projection=False, s=1):
        self._scatter(points, color, s)
        if not plot_projection:
            return

        proj_points = points.copy()
        proj_points[:, 2] = 0
        self._scatter(proj_points, color, s)

        dir = normalize(normalize(points).mean(axis=0).reshape(1, -1)).reshape(-1)
        arrow_dir = -np.sign(dir[2])
        arrow_length = abs(dir[2])
        self.ax.quiver(*dir, 0, 0, arrow_dir, length=arrow_length, color=color, arrow_length_ratio=0.2/arrow_length)

    def plot_dataset_embedding(self, dataset: SmartDataset, feature_extractor, count_by_class: Dict[int, int], default_count=100,
                               normalize_features=False, *args, **kwargs):
        COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        feature_extractor.eval().to(DEVICE)

        with torch.no_grad():
            for c, ss in dataset.by_class.items():
                count = count_by_class.get(c, default_count)
                if count == 0:
                    continue
                color = COLORS[c % len(COLORS)]
                samples = ss.random_subset(count).load()[0].to(DEVICE)
                embeddings = feature_extractor(samples)
                if normalize_features:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

                self.plot_cluster(embeddings.cpu().numpy(), color, *args, **kwargs)
