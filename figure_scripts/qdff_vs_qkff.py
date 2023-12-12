import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

kr_successes = []
qd_successes = []
Ms = [10]
log = True
eps = 1e-1
model = f"xyz_bias_{1-eps}"
bias = 1


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

fig, ax = plt.subplots(2, 2, figsize=(5, 5), sharey='row', sharex='col')


def plot_model(model, Ls):
    qd = pd.read_csv(os.path.join(path, f"../results/{model}/qdavidson_{Ls[0]}_{bias}.txt"))
    qd = qd[qd["fidelity"] > 1 - eps]

    kr_successes = []
    qd_successes = []
    for M in Ms:
        krylov = pd.read_csv(os.path.join(path, f"../results/{model}/qkrylov_{Ls[1]}_{M}_0.1_{bias}.txt"))
        kr_successes.append(krylov[(krylov["fidelity"] > 1 - eps)]) # & (krylov["qubits"] % 2 == 0)])

        qdavidson = pd.read_csv(os.path.join(path, f"../results/{model}/mr_qdavidson_{Ls[2]}_{M}_0.1_{bias}.txt"))
        qd_successes.append(qdavidson[(qdavidson["fidelity"] > 1 - eps)]) # & (qdavidson["qubits"] % 2 == 0) ])

    if model[:3]=="xxx":
        # xxx
        ax[0,0].plot(qd["qubits"], qd["num_states"], "-o", label="QD", color='k')
        ax[1,0].plot(qd["qubits"], qd["num_iters"], "-o", label="QD", color='k')

        for i, qd_success in enumerate(qd_successes):
            ax[0,0].plot(qd_success["qubits"], qd_success["num_states"], "-o", label=f"QD, M={Ms[i]}", color=colors[i])
            ax[1,0].plot(qd_success["qubits"], qd_success["num_iters"], "-o", label=f"QD, M={Ms[i]}", color=colors[i])

        for i, kr_success in enumerate(kr_successes):
            ax[0,0].plot(kr_success["qubits"], kr_success["num_states"], "-o", label=f"MRK, M={Ms[i]}", color=colors[i+1])
            ax[1,0].plot(kr_success["qubits"], kr_success["num_iters"], "-o", label=f"MRK, M={Ms[i]}", color=colors[i+1])


    else:
        # xyz
        ax[0,1].plot(qd["qubits"], qd["num_states"], "-o", label="QD", color='k')
        ax[1,1].plot(qd["qubits"], qd["num_iters"], "-o", label="QD", color='k')

        for i, qd_success in enumerate(qd_successes):
            ax[0,1].plot(qd_success["qubits"], qd_success["num_states"], "-o", label=f"QD, M={Ms[i]}", color=colors[i])
            ax[1,1].plot(qd_success["qubits"], qd_success["num_iters"], "-o", label=f"QD, M={Ms[i]}", color=colors[i])

        for i, kr_success in enumerate(kr_successes):
            ax[0,1].plot(kr_success["qubits"], kr_success["num_states"], "-o", label=f"MRK, M={Ms[i]}", color=colors[i+1])
            ax[1,1].plot(kr_success["qubits"], kr_success["num_iters"], "-o", label=f"MRK, M={Ms[i]}", color=colors[i+1])


plot_model(f"xxx_bias_{1-eps}", [13,11,11])
plot_model(f"xyz_bias_{1-eps}", [11,10,11])

ax[0,0].text(-.32,0.95,'(a)', transform=ax[0,0].transAxes, fontsize=12)
ax[1,0].text(-.32,0.95,'(b)', transform=ax[1,0].transAxes, fontsize=12)
ax[0,1].text(-.17,0.95,'(c)', transform=ax[0,1].transAxes, fontsize=12)
ax[1,1].text(-.17,0.95,'(d)', transform=ax[1,1].transAxes, fontsize=12)

ax[0,0].set_yscale('log')
ax[1,0].set_yscale('log')
ax[0,0].legend(loc="lower right", fontsize=8)

# # ax[0].set_xlabel("System size")
ax[1,0].set_xlabel("System size")
ax[1,1].set_xlabel("System size")
ax[1,1].set_xticks([5,10])

ax[0,0].set_ylabel("Krylov dimension")
ax[1,0].set_ylabel("Num iterations")

# plt.show()
plt.subplots_adjust(wspace=0.2, hspace=0.1)
plt.savefig(os.path.join(path, '../figures/qd_vs_mrk.png'), dpi=1000, transparent=False, bbox_inches='tight')
