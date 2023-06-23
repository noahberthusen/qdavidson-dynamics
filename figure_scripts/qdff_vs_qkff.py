import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

qdavidson = pd.read_csv(os.path.join(path, "../results/xxx/qdavidson.txt"), names=["qubits","num_iters","num_states","fidelity"])
qd_successes = qdavidson[qdavidson["fidelity"] > 1 - 1e-2]

krylov = pd.read_csv(os.path.join(path, "../results/xxx/qkrylov_10_0.1.txt"), names=["qubits","M","tau","num_iters","num_states","fidelity"])
kr_successes = krylov[krylov["fidelity"] > 1 - 1e-2]

# -----------------------------------------------------------------------------

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 1.2

# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = [(64, 83, 211), (221, 179, 16), (181, 29, 20), (0, 190, 255), (251, 73, 176), (0, 178, 93)]
# colors = [(239, 230, 69), (233, 53, 161), (0, 227, 255), (225, 86, 44), (83, 126, 255), (0, 203, 133)][::-1]
# colors = [(86, 100, 26), (192, 175, 251), (230, 161, 118), (0, 103, 138), (152, 68, 100), (94, 204, 171)]
colors = [(c[0]/255, c[1]/255, c[2]/255) for c in colors]
# colors = sns.color_palette("hls", 6)
# colors = sns.color_palette("Set2", 6)

fig, ax = plt.subplots(1, 1, figsize=(5.5,4))
ax.plot(qd_successes["qubits"], qd_successes["num_states"], "-o", label="QDavidson", color=colors[0])
ax.plot(kr_successes["qubits"], kr_successes["num_states"], "-o", label="Multi-reference Krylov", color=colors[1])

plt.legend(loc="lower right")
plt.yscale('log')
plt.xlabel("System size")
plt.ylabel("Krylov dimension")

# plt.show()
plt.savefig(os.path.join(path, '../figures/qd_vs_mrk.png'), dpi=1000, transparent=False, bbox_inches='tight')
