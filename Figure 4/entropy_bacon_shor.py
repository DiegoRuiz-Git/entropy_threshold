import stim
import numpy as np
import pymatching
import matplotlib.pyplot as plt
import matplotlib
import math
from numba import njit
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
plt.style.use('rgplot')
matplotlib.use("Agg")


# X and Z parity check matrix of Bacon Shor code
def Create_check_matrices(d):

    n = d**2

    Hz = np.zeros((d-1,n), dtype = int)

    for i in range(d-1):
        for j in range(d):
            Hz[i,i*d+j] = 1
            Hz[i,(i+1)*d+j] = 1

    Hx = np.zeros((d-1,n), dtype = int)

    for j in range(d-1):
        for i in range(d):
            Hx[j, i*d+j] = 1 
            Hx[j, i*d+j+1] = 1 
             
    return Hz,Hx


# Code capacity circuit Bacon Shor code
def Create_Circuit(d, p):
    
    Hz, Hx = Create_check_matrices(d)
    n = Hz.shape[1]
    
    circuit = stim.Circuit()
    
    circuit.append('R', range(n))
    
    for i in range(Hz.shape[0]):
        qubits = np.where(Hz[i])[0]
        mpp_string = []
        for qubit in qubits:
            mpp_string.append(stim.target_z(qubit))
            mpp_string.append('*')
        circuit.append('MPP',mpp_string[:-1])
        
    for i in range(Hx.shape[0]):
        qubits = np.where(Hx[i])[0]
        mpp_string = []
        for qubit in qubits:
            mpp_string.append(stim.target_x(qubit))
            mpp_string.append('*')
        circuit.append('MPP',mpp_string[:-1])
    
    circuit.append('X_ERROR', range(n), p)
    
    for i in range(Hz.shape[0]):
        qubits = np.where(Hz[i])[0]
        mpp_string = []
        for qubit in qubits:
            mpp_string.append(stim.target_z(qubit))
            mpp_string.append('*')
        circuit.append('MPP',mpp_string[:-1])
        circuit.append('DETECTOR', [stim.target_rec(-1), stim.target_rec(-2*(d-1)-1)])
        
    for i in range(Hx.shape[0]):
        qubits = np.where(Hx[i])[0]
        mpp_string = []
        for qubit in qubits:
            mpp_string.append(stim.target_x(qubit))
            mpp_string.append('*')
        circuit.append('MPP',mpp_string[:-1])
        circuit.append('DETECTOR', [stim.target_rec(-1), stim.target_rec(-2*(d-1)-1)])
        
    circuit.append('M', range(n))
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-n+i) for i in range(d)],0)
    
    return circuit


# stim simulation
p_list = np.logspace(-2,-1,10)
d_list = [5,7]

Logical_Error = np.zeros((len(d_list),len(p_list)))
Logical_Error_u = np.zeros((len(d_list),len(p_list)))

print('stim simulation')
for index_d, d in enumerate(d_list):
    for index_p, p in enumerate(p_list):
        print(f'd = {d} {index_p+1}/{len(p_list)}')
        circuit = Create_Circuit(d, p)
        dem = circuit.detector_error_model()
        matching = pymatching.Matching.from_detector_error_model(dem)

        batch_size = 1_000_000
        sampler = circuit.compile_detector_sampler()
        syndrome, actual_observables = sampler.sample(shots=batch_size, separate_observables=True)

        predicted_observables = matching.decode_batch(syndrome)
        num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))

        ler = num_errors/batch_size
        Logical_Error[index_d, index_p] = num_errors/batch_size
        Logical_Error_u[index_d, index_p] = 1.96*np.sqrt(1/batch_size*ler*(1-ler))


# determine crossing threshold
def spline_crossing(Logical_error, Logical_error_u, p_list, n_boot=1000):
    rng = np.random.default_rng(None)
    Logical_error = np.asarray(Logical_error)
    Logical_error_u = np.asarray(Logical_error_u)

    def fit_and_cross(y0, y1):
        s0 = UnivariateSpline(p_list, y0, w=1 / Logical_error_u[0], s=None)
        s1 = UnivariateSpline(p_list, y1, w=1 / Logical_error_u[1], s=None)
        return brentq(lambda x: s0(x) - s1(x), p_list[0], p_list[-1]), s0, s1

    p_cross, s5, s7 = fit_and_cross(Logical_error[0], Logical_error[1])

    crossings = np.empty(n_boot)
    for k in range(n_boot):
        y0 = rng.normal(Logical_error[0], Logical_error_u[0])
        y1 = rng.normal(Logical_error[1], Logical_error_u[1])
        try:
            crossings[k], _, _ = fit_and_cross(y0, y1)
        except ValueError:
            crossings[k] = np.nan

    p_cross_u = np.nanstd(crossings)

    return p_cross, p_cross_u

p_cross, p_cross_u = spline_crossing(Logical_Error, Logical_Error_u, p_list, n_boot=1000)


### entropy ###

# error entropy
def H_E(H, p):
    n = H.shape[1]

    H2 = -p*np.log2(p) - (1-p)*np.log2(1-p)

    return n*H2


# syndrome distribution
@njit(cache=True)
def _compute_column_ids(H):
    n_a, n = H.shape
    column_ids = np.zeros(n, dtype=np.int64)
    for j in range(n):
        s = 0
        for i in range(n_a):
            if H[i, j]:
                s |= (1 << i)
        column_ids[j] = s
    return column_ids


@njit(cache=True)
def distribution_S_jit_from_column_ids(column_ids, n_a, p):
    syndrome_space_size = 1 << n_a
    probas = np.zeros(syndrome_space_size, dtype=np.float64)
    new_probas = np.zeros(syndrome_space_size, dtype=np.float64)
    probas[0] = 1.0

    p_no_error = 1.0 - p

    for k in range(column_ids.shape[0]):
        h_id = column_ids[k]
        for s in range(syndrome_space_size):
            new_probas[s] = p_no_error * probas[s] + p * probas[s ^ h_id]

        probas, new_probas = new_probas, probas

    return probas


def distribution_S_jit(H, p):
    n_a, n = H.shape
    column_ids = _compute_column_ids(H)
    return distribution_S_jit_from_column_ids(column_ids, n_a, p)


# syndrome entropy
def H_S(H,p):
    pattern_proba = distribution_S_jit(H, p)
    return -np.sum(pattern_proba*np.log2(pattern_proba))


# joint distribution L,S
@njit(cache=True)
def _compute_column_ids(H):
    n_a, n = H.shape
    column_ids = np.zeros(n, dtype=np.int64)
    for j in range(n):
        s = 0
        for i in range(n_a):
            if H[i, j]:
                s |= (1 << i)
        column_ids[j] = s
    return column_ids


@njit(cache=True)
def _compute_obs_indicator(obs, n):
    out = np.zeros(n, dtype=np.uint8)
    for i in range(obs.shape[0]):
        out[obs[i]] = 1
    return out


@njit(cache=True)
def distribution_L_S_jit_from_column_ids_fast(column_ids, obs_indicator, n_a, p):
    m = 1 << n_a
    p_no = 1.0 - p

    p0 = np.zeros(m, dtype=np.float64)
    p1 = np.zeros(m, dtype=np.float64)
    q0 = np.zeros(m, dtype=np.float64)
    q1 = np.zeros(m, dtype=np.float64)

    p0[0] = 1.0

    for k in range(column_ids.shape[0]):
        h = column_ids[k]

        if obs_indicator[k] == 0:
            for s in range(m):
                sx = s ^ h
                q0[s] = p_no * p0[s] + p * p0[sx]
                q1[s] = p_no * p1[s] + p * p1[sx]
        else:
            for s in range(m):
                sx = s ^ h
                q0[s] = p_no * p0[s] + p * p1[sx]
                q1[s] = p_no * p1[s] + p * p0[sx]

        p0, q0 = q0, p0
        p1, q1 = q1, p1

    out = np.empty((2, m), dtype=np.float64)
    out[0, :] = p0
    out[1, :] = p1
    return out


def distribution_L_S_jit(H, obs, p):
    n_a, n = H.shape
    column_ids = _compute_column_ids(H)
    obs_indicator = _compute_obs_indicator(np.asarray(obs, dtype=np.int64), n)
    return distribution_L_S_jit_from_column_ids_fast(column_ids, obs_indicator, n_a, p)


# entropy of logical and syndrome
def H_L_S(H, obs, p):
    
    pattern_proba = distribution_L_S_jit(H, obs, p)

    entropy = 0

    for observable_index in range(pattern_proba.shape[0]):
        for syndrome_index in range(pattern_proba.shape[1]):
            if pattern_proba[observable_index,syndrome_index] != 0:
                entropy -= (
                    pattern_proba[observable_index,syndrome_index]*
                    np.log2(pattern_proba[observable_index,syndrome_index])
                )

    return entropy


# entropy of error knowing logical and syndrome
def H_E_knowing_L_S(H, obs, p):
    return H_E(H,p) - H_L_S(H,obs,p)


# parity check matrix
def H(d):
    n = d**2

    H = np.zeros((d-1,n), dtype = int)

    for i in range(d-1):
        for j in range(d):
            H[i,i*d+j] = 1
            H[i,(i+1)*d+j] = 1
            
    return H


# observable
def obs(d):
    return np.arange(d)


# distance 5
d = 5

H_S_d5 = []
for p in p_list:
    H_S_d5.append(H_S(H(d),p))

H_E_knowing_L_S_d5 = []
for p in p_list:
    H_E_knowing_L_S_d5.append(H_E_knowing_L_S(H(d),obs(5),p))


# distance 7
d = 7

H_S_d7 = []
for index,p in enumerate(p_list):
    print(index, end='\r')
    H_S_d7.append(H_S(H(d),p))

H_E_knowing_L_S_d7 = []
for i, p in enumerate(p_list):
    H_E_knowing_L_S_d7.append(H_E_knowing_L_S(H(d), obs(d), p))


# independant syndrome entropy variation
syndrome_entropies_indep = np.zeros(len(p_list))

for index_p, p in enumerate(p_list):   
    q = (1-(1-2*p)**14)/2
    r = (1-(1-2*p)**10)/2
    syndrome_entropies_indep[index_p] = 6*(-q*np.log2(q)-(1-q)*np.log2(1-q)) - 4*(-r*np.log2(r)-(1-r)*np.log2(1-r))


# plot
plt.figure(figsize = (2.5,2))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.plot([p_cross,p_cross], [0.6,3.5],'--',c = 'k')
plt.plot(p_list, H_E(H(d=7),p_list) - H_E(H(d=5),p_list),'-', c = colors[1])
plt.plot(p_list, np.array(H_S_d7) - np.array(H_S_d5),'-', c = colors[0])
plt.plot(p_list, syndrome_entropies_indep,'--', c = colors[0])
plt.plot(p_list, np.array(H_E_knowing_L_S_d7) - np.array(H_E_knowing_L_S_d5),'-', c = colors[2])

plt.text(0.03, 1, r"$p_\times$", fontsize = 9)

plt.tick_params(axis='both', which='major', labelsize=7)
plt.tick_params(axis='both', which='minor', label1On=False, label2On=False)
plt.xlim((1e-2,1e-1))

plt.xlabel(r'$p$', fontsize = 9)
plt.ylabel(r'$\Delta_{5 \rightarrow 7} H$', fontsize = 9)
plt.loglog()
plt.title('Bacon-Shor code', fontsize = 7)
plt.savefig('entropy_bacon_shor.pdf')