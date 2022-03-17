"""
This file defines some utilities functions for plotting.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from typing import List

from lif import LIF
from utils_analysis import Response

def plot_clusters3d(x, y):
    """
    Plots the provided clustered data in 3D.

    Arguments:
    - x: ndarray[nxd] where d >= 3
        at least 3D data
    - y: ndarray[nx1]
        cluster IDs of each sample
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,0], x[:,1], x[:,2], c=y)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

def plot_clusters2d(x, y):
    """
    Plots the provided clustered data in 2D.

    Arguments:
    - x: ndarray[nxd] where d >= 3
        at least 3D data
    - y: ndarray[nx1]
        cluster IDs of each sample
    """
    plt.scatter(x[:,0], x[:,1], c=y)

def visaulize_response(response: Response):
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex='col')
    ax0.plot(response.time, response.stimulus)
    ax0.set_ylabel('stimulus')

    ax1.plot(response.time, response.spikes)
    ax1.set_ylabel('spikes')

    ax1.set_xlabel('time (ms)')


def plot_params(neurons: List[LIF]):
    """
    Plots the parameters of the provided list of LIFs on histograms.

    Arguments:
    - neurons: List[LIF]
    """
    fig, axs = plt.subplots(2,3,sharex=False)

    plt.tight_layout()

    # axs[0, 0].hist([n.c_m for n in neurons], alpha=0.7)
    # axs[0, 0].set_xlim([10,500])
    # axs[0, 0].set_title('capacitance')

    # axs[0, 2].hist([n.g_l for n in neurons], alpha=0.7)
    # axs[0, 2].set_xlim([1,100])
    # axs[0, 2].set_title('conductance')

    # axs[0, 0].hist([n.c_m for n in neurons], alpha=0.7)
    # axs[0, 0].set_xlim([10,500])
    # axs[0, 0].set_title('capacitance')

    # axs[0, 2].hist([n.g_l for n in neurons], alpha=0.7)
    # axs[0, 2].set_xlim([1,100])
    # axs[0, 2].set_title('conductance')

    # axs[0, 1].remove()

    # axs[1, 0].hist([n.v_l for n in neurons], alpha=0.7)
    # axs[1, 0].set_xlim([-100,-75])
    # axs[1, 0].set_title('rest voltage')

    # axs[1,1].remove()

    # axs[1, 2].hist([n.v_th for n in neurons], alpha=0.7)
    # axs[1, 2].set_xlim([-100,50])
    # axs[1, 2].set_title('threshold')

    axs[1, 0].hist([n.k_j[0] for n in neurons], alpha=0.7)
    axs[1, 0].hist([n.k_j[1] for n in neurons], alpha=0.7)
    axs[1, 0].set_xlim([1e-9, 1])
    axs[1, 0].set_title('k_j')

    axs[1, 1].hist([n.f_j[0] for n in neurons], alpha=0.7)
    axs[1, 1].hist([n.f_j[1] for n in neurons], alpha=0.7)
    axs[1, 1].set_xlim([-200, 200])
    axs[1, 1].set_title('f_j')

    axs[1, 2].hist([n.b_j[0] for n in neurons], alpha=0.7)
    axs[1, 2].hist([n.b_j[1] for n in neurons], alpha=0.7)
    axs[1, 2].set_xlim([-1,1])
    axs[1, 2].set_title('b_j')

def visualize_stc(pca: PCA, dt: float):
    n_components = pca.n_components_
    n_features = pca.n_features_
    explained_var = pca.explained_variance_

    time = np.arange(0.0, n_features * dt, dt)
    for i in range(n_components):
        plt.plot(time, pca.components_[i, :], label = f"PC{i}: {round(explained_var[i],2)}")

def visualize_paramvar(neurons: List[LIF], projections):
    """
    Plots the variance in each of the first three PCs with different parameters.

    Arguments:
    - neurons: List[LIF]
    - projections
    """
    fig, axs = plt.subplots(2,3,sharex=False)

    plt.tight_layout()

    # axs[0, 0].scatter([n.c_m for n in neurons], projections[:,0])
    # axs[0, 0].scatter([n.c_m for n in neurons], projections[:,1])
    # axs[0, 0].scatter([n.c_m for n in neurons], projections[:,2])
    # axs[0, 0].set_xlim([10,500])
    # axs[0, 0].set_title('capacitance')

    # axs[0, 2].scatter([n.g_l for n in neurons], projections[:,0])
    # axs[0, 2].scatter([n.g_l for n in neurons], projections[:,1])
    # axs[0, 2].scatter([n.g_l for n in neurons], projections[:,2])
    # axs[0, 2].set_xlim([1,100])
    # axs[0, 2].set_title('conductance')

    # axs[0, 1].remove()

    # axs[1, 0].scatter([n.v_l for n in neurons], projections[:,0])
    # axs[1, 0].scatter([n.v_l for n in neurons], projections[:,1])
    # axs[1, 0].scatter([n.v_l for n in neurons], projections[:,2])
    # axs[1, 0].set_xlim([-100,-75])
    # axs[1, 0].set_title('rest voltage')

    # axs[1,1].remove()

    # axs[1, 2].scatter([n.v_th for n in neurons], projections[:,0])
    # axs[1, 2].scatter([n.v_th for n in neurons], projections[:,1])
    # axs[1, 2].scatter([n.v_th for n in neurons], projections[:,2])
    # axs[1, 2].set_xlim([-100,50])
    # axs[1, 2].set_title('threshold')

    axs[1, 0].scatter([n.k_j[0] for n in neurons], projections[:,0])
    axs[1, 0].scatter([n.k_j[0] for n in neurons], projections[:,1])
    axs[1, 0].scatter([n.k_j[0] for n in neurons], projections[:,2])
    # ax1[2, 0].hist([n.k_j[1] for n in neurons], alpha=0.7)
    axs[1, 0].set_xlim([1e-9, 1])
    axs[1, 0].set_title('k_j')

    axs[1, 1].scatter([n.f_j[0] for n in neurons], projections[:,0])
    axs[1, 1].scatter([n.f_j[0] for n in neurons], projections[:,1])
    axs[1, 1].scatter([n.f_j[0] for n in neurons], projections[:,2])
    axs[1, 1].set_xlim([-200, 200])
    axs[1, 1].set_title('f_j')

    axs[1, 2].scatter([n.b_j[0] for n in neurons], projections[:,0])
    axs[1, 2].scatter([n.b_j[0] for n in neurons], projections[:,1])
    axs[1, 2].scatter([n.b_j[0] for n in neurons], projections[:,2])
    axs[1, 2].set_xlim([-1,1])
    axs[1, 2].set_title('b_j')