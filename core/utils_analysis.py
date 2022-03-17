"""
This file defines some utilities functions for analysis.
"""

from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List, NamedTuple
from lif import LIF, LIFParams


class Response(NamedTuple):
    """Structure containing related information about a single response"""
    time: np.ndarray # array of times in msec
    stimulus: np.ndarray # array of the I_ext
    spikes: np.ndarray # binary array of spikes

def gen_neuron(dt: float)->LIF:
    """
    Returns randomly initializes neuron.

    Arguments:
    - dt: float
        timestep duration to use in forward euler

    Returns:
    - LIF
        LIF neuron with random initialization
    """
    return LIF(
        LIFParams(
            dt=dt,
            c_m = random.uniform(50, 500),
            g_l=random.uniform(1, 20),
            v_l = random.uniform(-100, -75),
            v_th = random.uniform(-75, 20),
            k_j = np.random.uniform(1e-9, 1, 2),
            f_j = np.random.uniform(-200, 200, 2),
            b_j = np.random.uniform(-1, 1, 2)
        )
    )

def stim_neuron(neuron: LIF, time_duration: float)->Response:
    """
    Stimulates neuron with white noise stimulus and computes response.

    Arguments:
    - neuron: LIF
        the neuron to stimulate
    - time_duration: float
        the length (in msec) of stimulus and simulation
    
    Returns:
    - response
    """
    time = np.arange(0, time_duration, neuron.dt)
    stim = 1e4 * np.random.randn(len(time))
    spikes = neuron.forward(stim)
    return Response(
        time=time, stimulus=stim, spikes=spikes
    )

def compute_sts(response: Response, trigger_duration: float):
    """
    Compute a list of spike-triggered (ST) stimuli from response.

    Arguments:
    - response: Response
        stimulus-response pair to compute STs from
    - trigger_duration: float
        the length (in msec) of each spike-triggered stimulus
    
    Returns:
    - List
        list of Response objects, each comprising a ST
    """
    time = response.time
    stim = response.stimulus
    spikes = response.spikes
    dt = time[1] - time[0]
    nsteps = int(trigger_duration / dt)

    sts = []
    for i in range(nsteps, len(time)):
        if spikes[i]:
            sts.append(
                Response(
                    time = time[i - nsteps:i+1],
                    stimulus = stim[i-nsteps:i+1],
                    spikes = spikes[i-nsteps: i+1]
                )
            )
    return sts

def compute_pcs(k: int, responses: List[Response], trigger_duration: float):
    """
    Fit a PCA object to STs from provided stimulus-response pairs.

    Arguments:
    - k: int
        number of principal components to preserve
    - responses: List[Response]
        list of stimulus-response pairs to compute STs for
    - trigger_duration: float
        duration to use for STs

    Returns:
    a fit PCA object from sklearn
    and the list of STAs
    """
    # Compute STs
    sts = []
    for response in responses:
        sts_r = compute_sts(response, trigger_duration)
        sts_array = np.stack([r.stimulus for r in sts_r])
        sts.append(np.mean(sts_array, axis=0))
        # sts.extend(np.mean(compute_sts(response, trigger_duration), axis=0))
    
    # Compute pca
    pca = PCA(n_components=k)
    # sts_array = np.array([r.stimulus for r in sts])
    pca.fit(np.stack(sts).reshape((len(responses), -1)))
    return pca, np.stack(sts)

def project_sta(response: Response, pca: PCA, trigger_duration: float):
    """
    Compute STA projection onto principal components

    Arguments:
    - response: Response
        stimulus-response pairs to compute STA from
    - pca: PCA
        a fit PCA object
    - trigger_duration: float
        duration to use for STs
    
    Returns:
    - ndarray
        NumPY array 1xk
    """
    # Compute STA
    sts = compute_sts(response, trigger_duration)
    sts_array = np.array([r.stimulus for r in sts])
    sta = np.mean(sts_array, axis=0)
    # Project
    return pca.transform(sta.reshape(1, -1))

def cluster(x, k: int):
    """
    Cluster provided data using KMeans clustering
    
    Arguments:
    - x: ndarray [nxd]
        data to cluster where there are n samples
    - k: int
        number of clusters to estimate
    
    Returns
    - ndarray [nx1]
        a list of cluster IDs for each sample
    """
    kmeans = KMeans(n_clusters=k)
    return kmeans.fit_predict(x)