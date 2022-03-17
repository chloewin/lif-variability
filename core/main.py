"""
This file contains the main code for analysis
"""

from itertools import compress
import utils_analysis as uta
import utils_plotting as utp
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

dt = 0.5
time_duration = 3000
trigger_duration = 100
k = 3
n = 500
k_clusters = 1

ntrigger = int(trigger_duration / dt)

alphabets = ["A", "B", "C", "D", "E"]
representative = {random.randint(0, n - 1):alphabets[i] for i in range(5)}

responses = []
skipped = 0
lifs = []
for i in tqdm(range(n)):
    lif = uta.gen_neuron(dt = dt)
    lifs.append(lif)
    r = uta.stim_neuron(neuron=lif, time_duration=time_duration)
    responses.append(r)
    if i in representative:
        utp.visaulize_response(r)
        plt.savefig(f"./../images/response{representative[i]}.jpg")
        plt.close()

    if (sum(r.spikes) == 0):
        skipped += 1
        continue

pca = uta.compute_pcs(k, responses, trigger_duration)

projections = np.zeros((n, k))
num_spikes = []
for i in tqdm(range(n)):
    r = responses[i]
    if (sum(r.spikes[ntrigger:]) == 0):
        r.spikes[ntrigger:] = np.ones(len(r.spikes) - ntrigger)
    projections[i,:] = uta.project_sta(r, pca, trigger_duration)
    num_spikes.append(sum(r.spikes))

projections = projections / projections.max()
clusters = uta.cluster(projections, k_clusters)
utp.plot_clusters3d(projections, clusters)
plt.savefig("./../images/3d-clusters.jpg")
plt.close()

utp.plot_clusters2d(projections, clusters)
for i in representative:
    plt.text(projections[i, 0], projections[i, 1], representative[i])
plt.savefig("./../images/2d-clusters.jpg")
plt.close()

utp.plot_params(list(compress(lifs, clusters)))
plt.savefig("./../images/parameters-cluster1.jpg")
plt.close()

utp.plot_params(list(compress(lifs, [not c for c in clusters])))
plt.savefig("./../images/parameters-cluster2.jpg")
plt.close()

utp.visualize_stc(pca, dt)
plt.savefig("./../images/stc.jpg")
plt.close()

utp.visualize_paramvar(lifs, projections)
plt.savefig("./../images/paramvar.jpg")
plt.close()

print(f"skipped {skipped}")