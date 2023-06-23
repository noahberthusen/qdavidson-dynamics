import numpy as np
import copy
from scipy.linalg import eigh
from scipy.sparse.linalg import expm, expm_multiply
from scipy.sparse import csc_matrix, lil_matrix, eye
import datetime
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Command line arguments
    parser.add_argument("--L", type=int, default=5, help="System size")
    parser.add_argument("--q", type=int, default=1, help="Whether or not to run qDavidson (0/1)")
    parser.add_argument("--m", type=int, default=10, help="Krylov dimension")
    parser.add_argument("--t", type=float, default=0.5, help="Trotter step size")
    args = parser.parse_args()
    num_qubits = args.L
    run_davidson = args.q
    M = args.m
    tau = args.t

    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    def heisenberg(L):
        def FlipFlop(n, i, j):
            v = list(format(n, '0{}b'.format(L)))
            if (v[i] != '0' and v[j] != '1'):
                v[i] = '0'
                v[j] = '1'
                return int(''.join(v), 2)
            else:
                return -1
            
        sprs = lil_matrix((2**L, 2**L), dtype=np.int8)
        for i in range(L-1): 
            for j in range(2**L):
                h = FlipFlop(j, i, i+1)
                if (h != -1):
                    sprs[j, h] = 2
                    sprs[h, j] = 2

                v = lambda k: 1-2*int(format(j, '0{}b'.format(L))[k])
                sprs[j, j] += v(i) * v(i+1)

        # bias term
        for i in range(L):
            for j in range(2**L):
                sprs[j, j] += v(i)

        return sprs.tocsc()

    def correction_state(ham, energy, state, tau=0.0001):
        #op = expm(-tau*(ham-energy))
        #correction_state = op @ state
        correction_state = ham @ state - energy*state
        return correction_state / np.linalg.norm(correction_state)

    # def eff_ham(ham, basis_set): 
    #     eff_H = np.eye(len(basis_set), dtype=complex)
    #     for i in range(len(basis_set)): 
    #         for j in range(len(basis_set)): 
    #             eff_H[i][j] = basis_set[i].conj().T.dot(ham.dot(basis_set[j]))
    #     return eff_H    

    # def eff_overlap(basis_set): 
    #     eff_S = np.eye(len(basis_set), dtype=complex)
    #     for i in range(len(basis_set)): 
    #         for j in range(len(basis_set)): 
    #             eff_S[i][j] = basis_set[i].conj().dot(basis_set[j])
    #     return eff_S

    def eff_ham(ham, basis_set): 
        eff_H = np.eye(len(basis_set), dtype=complex)
        for i in range(len(basis_set)): 
            for j in range(i,len(basis_set)): 
                overlap = np.conj(basis_set[i]).dot(ham.dot(basis_set[j]))
                eff_H[i][j] = overlap
                if (i != j):
                    eff_H[j][i] = np.conj(overlap)
        return eff_H   

    def eff_overlap(basis_set): 
        eff_S = np.eye(len(basis_set), dtype=complex)
        for i in range(len(basis_set)): 
            for j in range(i,len(basis_set)): 
                overlap = basis_set[i].conj().dot(basis_set[j])
                eff_S[i][j] = overlap
                if (i != j):
                    eff_S[j][i] = np.conj(overlap)
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
        b = np.array([correction_vec.conj().T.dot(basis_set[i]) for i in range(len(basis_set))])
        if np.all(np.round(eff_S, 8) == np.eye(len(basis_set))):
            return np.linalg.norm(b) < tol
        else:
            return np.linalg.norm(np.linalg.pinv(eff_S).dot(b)) < tol    

    def qdavidson(ham, initial_basis_set, num_iter, tol=0.5): 
        basis_set = copy.deepcopy(initial_basis_set)
        for i in range(num_iter): 
            evals, estates, residue_vals, basis_set, eff_H, eff_S = qdavidson_iter(ham, basis_set, tol)
        return evals, estates, residue_vals, basis_set, eff_H, eff_S

    def QDFFEvolve(S_inv, H, t, basis, init):
        ct = expm(-1j * t * S_inv @ H) @ init
        return sum([ct[i] * basis[i] for i in range(len(ct))])


    # --------------------------------------------------------

    # run_davidson = False
    # num_qubits = 11  # calculate up to num_qubits qubits
    epsilon = 0.01  # we want the fidelity with the exact state to be at least this at time = tf

    tf = 10
    ts = np.linspace(0, tf, 200)

    # M = 8
    # tau = 0.1
    eff_S_eps = 1e-5  # cutoff values in eff_S less than this value

    cs = list("01010101010101010101010101010101010101010101010101")


    # ---------------------------------------------------------


    if (run_davidson):
        filename = "./results/heisenberg/" + f"qdavidson_{num_qubits}.txt"
        file = open(os.path.join(path, filename), 'w')
        file.write("qubits,num_iters,num_states,fidelity\n")

        for i in range(num_qubits, num_qubits+1):
            c = cs[0:i]
            UnitVector = lambda c: eye(2**i)[c]
            init = np.zeros(2**num_qubits)
            init[int(''.join(c), 2)] = 1
            basis_set = [init]
            ham = heisenberg(i)
            exact_final_te = expm_multiply(-1j * ham * ts[-1], init)

            num_iters = 1
            while (True):
                _, _, _, basis_set, eff_H, eff_S = qdavidson(ham, basis_set, 1, tol=0.0001)
                c0 = np.zeros(len(basis_set))
                c0[0] = 1

                qdff_final_te = QDFFEvolve(np.linalg.inv(eff_S), eff_H, ts[-1], basis_set, c0)

                fidelity = np.abs(np.conj(qdff_final_te) @ exact_final_te)**2
                # print(f"{i},{num_iters},{len(basis_set)},{fidelity}")
                file.write(f"{i},{num_iters},{len(basis_set)},{fidelity}\n")
                if (fidelity > 1-epsilon):
                    break

                num_iters += 1
        file.close()
    else:
        filename = "./results/heisenberg/" + f"qkrylov_{num_qubits}_{M}_{tau}.txt"
        file = open(os.path.join(path, filename), 'w')
        file.write("qubits,M,tau,num_iters,num_states,fidelity\n")

        for i in range(num_qubits, num_qubits+1):
            c = cs[0:i]
            UnitVector = lambda c: np.eye(2**i)[c]
            init = UnitVector(int(''.join(c), 2))
            added_indices = []
            ham = heisenberg(i)
            exact_final_te = expm_multiply(-1j * ham * ts[-1], init)

            expm_est = expm(-1j * tau * ham) # should be trotterized really, eventually
            
            basis_set = [init]
            for j in range(1, M):
                new_states = []
                new_states.append(expm_multiply(-1j * j * tau * ham, basis_set[-1]))
                basis_set.extend(new_states)

            num_ref_states = 1
            while (True):
                c0 = np.zeros(len(basis_set))
                c0[0] = 1

                eff_S = eff_overlap(basis_set)
                eff_H = eff_ham(ham, basis_set)
                
                U, D, Vh = np.linalg.svd(eff_S)
                D[np.abs(D) < eff_S_eps] = 0
                D_inv = [1/d if d > 0 else 0 for d in D]
                S_inv = Vh.conj().T @ np.diag(D_inv) @ U.conj().T
                
                qdff_final_te = QDFFEvolve(S_inv, eff_H, ts[-1], basis_set, c0)
                fidelity = np.abs(np.conj(qdff_final_te) @ exact_final_te)**2
                # print(f"{i},{M},{tau},{num_ref_states},{len(basis_set)},{fidelity}")

                file.write(f"{i},{M},{tau},{num_ref_states},{len(basis_set)},{fidelity}\n")
                if (fidelity > 1 - epsilon):
                    break       

                # should take second most probable if the first one is already in there
                xs = np.eye(2**i)
                # probs = [ np.abs(np.conj(x) @ ref_states[-1])**2 for x in xs]
                # ref_states.append(xs[np.argmax(probs)])
                sorted_probs_inds = np.argsort([np.abs(np.conj(x) @ basis_set[-1])**2 for x in xs])[::-1]
                for k in range(len(sorted_probs_inds)):
                    if not (sorted_probs_inds[k] in added_indices):
                        num_ref_states += 1
                        basis_set.append(xs[sorted_probs_inds[k]])
                        for j in range(1, M):
                            new_states = []
                            new_states.append(expm_multiply(-1j * j * tau * ham, basis_set[-1]))
                            basis_set.extend(new_states)
                        added_indices.append(sorted_probs_inds[k])
                        break
        file.close()

