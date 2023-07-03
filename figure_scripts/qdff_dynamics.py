import numpy as np
import copy
from scipy.linalg import eigh
from scipy.sparse.linalg import expm, expm_multiply
import matplotlib.pyplot as plt
import os
import pandas as pd


full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

# pauli matrices
pauli = np.array([np.array([[1,0],[0,1]]), np.array([[0,1],[1,0]]), np.array([[0,-1.j],[1.j,0]]), np.array([[1,0],[0,-1]])])
pauli_tensor = np.array([[np.kron(pauli[i], pauli[j]) for i in range(4)] for j in range(4)])

# building operators 
def kronecker_pad(matrix, num_qubits, starting_site): 
    ''' pads a 1- or 2- local operator with identities on other sites to get 2^n by 2^n matrix '''
    kron_list = [np.eye(2) for i in range(num_qubits)]    
    kron_list[starting_site] = matrix
    if matrix.shape[0] == 4: 
        del kron_list[starting_site+1]
    
    padded_matrix = kron_list[0]
    for i in range(1, len(kron_list)):
        padded_matrix = np.kron(kron_list[i], padded_matrix)    
    return padded_matrix

# models
def heisenberg(num_qubits, bias_coeff=1.0, x_hopping_coeff=1.0, y_hopping_coeff=1.0, z_hopping_coeff=1.0): 
    terms = []
    # for i in range(num_qubits): 
    #     bias = bias_coeff*kronecker_pad(pauli[3], num_qubits, i)
    #     terms.append(bias)
        
    for i in range(num_qubits-1): 
        z_hop = z_hopping_coeff*kronecker_pad(pauli_tensor[(3,3)], num_qubits, i)
        terms.append(z_hop)
        y_hop = y_hopping_coeff*kronecker_pad(pauli_tensor[(2,2)], num_qubits, i)
        terms.append(y_hop)
        x_hop = x_hopping_coeff*kronecker_pad(pauli_tensor[(1,1)], num_qubits, i)
        terms.append(x_hop)
    
    return sum(terms)

def correction_state(ham, energy, state, tau=0.0001):
    #op = expm(-tau*(ham-energy))
    #correction_state = op @ state
    correction_state = ham @ state - energy*state
    return correction_state / np.linalg.norm(correction_state)

def eff_ham(ham, basis_set): 
    eff_H = np.eye(len(basis_set), dtype=complex)
    for i in range(len(basis_set)): 
        for j in range(len(basis_set)): 
            eff_H[i][j] = basis_set[i].conj().T @ ham @ basis_set[j]
    return eff_H    

def eff_overlap(basis_set): 
    eff_S = np.eye(len(basis_set), dtype=complex)
    for i in range(len(basis_set)): 
        for j in range(len(basis_set)): 
            eff_S[i][j] = basis_set[i].conj().T @ basis_set[j]
    return eff_S
    
def qdavidson_iter(ham, basis_set, tol=0.5):
    num_basis = len(basis_set)
    eff_H = eff_ham(ham, basis_set)
    eff_S = eff_overlap(basis_set)        
    evals, evecs = eigh(eff_H, eff_S)
    estates = [np.array(sum([evecs[:,i][j] * basis_set[j] for j in range(num_basis)])) for i in range(num_basis)]
    new_basis_set = copy.deepcopy(basis_set)
    residue_vals = []
    for i in range(num_basis): 
        val = np.linalg.norm((ham @ estates[i]) - (evals[i] * estates[i]))
        residue_vals.append(val)
        if val > tol: 
            state = correction_state(ham, evals[i], estates[i])            
            if linear_independence(state, new_basis_set, eff_S, tol): 
                '''
                eff_S = np.pad(eff_S, ((0, 1), (0, 1)), mode='constant')
                for i in range(len(new_basis_set)):
                    overlap = state.conj().T @ new_basis_set[i]
                    eff_S[i][len(new_basis_set)] = overlap
                    eff_S[len(new_basis_set)][i] = overlap
                '''
                new_basis_set.append(state)
                eff_S = eff_overlap(new_basis_set)
                eff_H = eff_ham(ham, new_basis_set)
            
    return evals, estates, residue_vals, new_basis_set, eff_H, eff_S

def linear_independence(correction_vec, basis_set, eff_S, tol=0.01): 
    b = np.array([correction_vec.conj().T @ basis_set[i] for i in range(len(basis_set))])
    return np.linalg.norm(np.linalg.pinv(eff_S) @ b) < tol    

def qdavidson(ham, initial_basis_set, num_iter, tol=0.5): 
    basis_set = copy.deepcopy(initial_basis_set)
    for i in range(num_iter): 
        evals, estates, residue_vals, basis_set, eff_H, eff_S = qdavidson_iter(ham, basis_set, tol)
    return evals, estates, residue_vals, basis_set, eff_H, eff_S

def QDFFEvolve(S, H, t, basis, init):
    ct = expm(-1j * t * np.linalg.inv(S) @ H) @ init
    return sum([ct[i] * basis[i] for i in range(len(ct))])

# ------------------------------------------------------------

num_qubits = 6
krylov_dims = [5, 7, 8]

tf = 10
ts = np.linspace(0, tf, 200)

c = list("010101") 
UnitVector = lambda c: np.eye(2**num_qubits)[c]
init = UnitVector(int(''.join(c), 2))

ham = heisenberg(num_qubits, x_hopping_coeff=1, y_hopping_coeff=2, z_hopping_coeff=3)

df = pd.read_csv(os.path.join(path, "../results/xyz/qdavidson_9.txt"))
qd_successes = df[df["qubits"] == num_qubits]

# ------------------------------------------------------------


exact_te = [expm_multiply(-1j * ham * t, init) for t in ts]
qdff_tes = []
for krylov_dim in krylov_dims:
    basis_set = [init]
    _, _, _, basis_set, eff_H, eff_S = qdavidson(ham, basis_set, krylov_dim, tol=0.0001)

    c0 = np.zeros(len(basis_set))
    c0[0] = 1

    qdff_tes.append([QDFFEvolve(eff_S, eff_H, t, basis_set, c0) for t in ts])


exact_auto = [np.abs(np.conj(init) @ psi)**2 for psi in exact_te]
qdff_autos = []
fidelities = []
for qdff_te in qdff_tes:
    qdff_autos.append([np.abs(np.conj(init) @ psi)**2 for psi in qdff_te])
    fidelities.append([np.abs(np.conj(psi) @ phi)**2 for psi,phi in zip(exact_te, qdff_te)])

# ------------------------------------------------------------

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

height_ratios = np.ones(len(krylov_dims)+1)
height_ratios[-1] = 3
fig, ax = plt.subplots(len(krylov_dims)+1, 1, figsize=(5,7), gridspec_kw={'height_ratios': height_ratios}, sharex=True)


for i, qdff_auto in enumerate(qdff_autos):
    # ax[i].set_title(f"{krylov_dims[i]}")
    ax[i].plot(ts, exact_auto, c='k')
    ax[i].plot(ts, qdff_auto, c=colors[i])


    ax[i].tick_params(axis='x', labelsize=10)
    ax[i].tick_params(axis='y', labelsize=10)

    ax[-1].plot(ts, fidelities[i], c=colors[i], label=f"{krylov_dims[i]}")


handles,labels = ax[-1].get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]
ax[-1].legend().get_frame().set_linewidth(1)
ax[-1].legend(handles, labels, loc='lower left', framealpha=0.8)
ax[-1].tick_params(axis='x', labelsize=10)
ax[-1].tick_params(axis='y', labelsize=10)

ax[1].set_ylabel("$C_n(\\tau)$", fontsize=12)
ax[-1].set_xlabel("Time, $t$", fontsize=12)
ax[-1].set_ylabel("Fidelity, $\mathcal{F}$", fontsize=12)
# ax[-1].set_yscale('log')
# plt.tight_layout()

cs = [(0,0,0) for _ in qd_successes["num_states"]]
# for i,kylov_dim in enumerate(krylov_dims):
#     cs[krylov_dim] = colors[i]
ins = ax[-1].inset_axes([0.75, 0.60, 0.2, 0.32])
ins.scatter(qd_successes["num_states"], qd_successes["fidelity"], c=cs, marker='o', s=8)
ins.tick_params(axis='x', labelsize=8)
ins.tick_params(axis='y', labelsize=8)
ins.set_ylabel("$\mathcal{F}$", fontsize=10)
ins.set_xlabel("Krylov dim.", fontsize=10)

plt.savefig(os.path.join(path, '../figures/qdff_auto.png'), dpi=1000, transparent=False, bbox_inches='tight')
