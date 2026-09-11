import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib
from numba import njit, prange, get_num_threads, get_thread_id

p = 0.01 # physical error

n_q = 9  # number of data qubits


def Create_parity_check(Clifford_deformed):
    H = np.zeros((n_q-1, 2*n_q), dtype = int)

    # CSS surface code parity check
    H[0,[0,1]] = 1
    H[1,[1,2,4,5]] = 1
    H[2,[3,4,6,7]] = 1
    H[3,[7,8]] = 1
    H[4,[0+n_q,1+n_q,3+n_q,4+n_q]] = 1
    H[5,[2+n_q,5+n_q]] = 1
    H[6,[3+n_q,6+n_q]] = 1
    H[7,[4+n_q,5+n_q,7+n_q,8+n_q]] = 1

    for qubit in range(n_q):
        if Clifford_deformed[qubit] == 1:
            for check in H:
                # turn X check into Z
                if check[qubit] == 1:
                    check[qubit] = 0
                    check[qubit + n_q] = 1
                # turn Z check into X
                elif check[qubit + n_q] == 1:
                    check[qubit + n_q] = 0
                    check[qubit] = 1
        if Clifford_deformed[qubit] == 2:
            for check in H:
                # turn Z check into Y
                if check[qubit + n_q] == 1:
                    check[qubit] = 1
            
    return H


def Create_logicals(Clifford_deformed):
    # observable surface code
    obs = np.array([
        [0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0],
        [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    for qubit in range(n_q):
        if Clifford_deformed[qubit] == 1:
            for logical in obs:
                # turn X logical into Z
                if logical[qubit] == 1:
                    logical[qubit] = 0
                    logical[qubit + n_q] = 1
                # turn Z logical into X
                elif logical[qubit + n_q] == 1:
                    logical[qubit + n_q] = 0
                    logical[qubit] = 1
        if Clifford_deformed[qubit] == 2:
            for logical in obs:
                # turn X logical into Y
                if logical[qubit] == 1:
                    logical[qubit + n_q] = 1

    return obs



@njit
def popcount(x):
    c = 0
    while x > 0:
        x &= x - 1
        c += 1
    return c

@njit
def parity(x):
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1

@njit(parallel=True)
def compute_core(n_a, n_q, H_X_ints, H_Z_ints, proba_error, N, mask_obs_0, mask_obs_1):
    num_threads = get_num_threads()
    pattern_proba_local = np.zeros((num_threads, 2, 2, 1 << n_a), dtype=np.float64)
    
    # Pre-compute the mask for the lower n_q bits
    q_mask = (1 << n_q) - 1

    for i in prange(N):
        tid = get_thread_id()
        
        # Original logic: bits[:, 0:n_q] is the first half, bits[:, n_q:n] is the second
        b_first = i & q_mask
        b_second = i >> n_q
        
        # --- Error Counting (Matching your original nb_errors_X/Y/Z) ---
        # nb_errors_X = (bits[:,0:n_q] & (bits[:,n_q:2*n_q]^1))
        nx = popcount(b_first & ~b_second)
        # nb_errors_Y = (bits[:,0:n_q] & bits[:,n_q:2*n_q])
        ny = popcount(b_first & b_second)
        # nb_errors_Z = ((bits[:,0:n_q]^1) & bits[:,n_q:2*n_q])
        nz = popcount(~b_first & b_second & q_mask)
        
        prob = proba_error[nx, ny, nz]
        
        # --- Syndrome Calculation ---
        # syndromes = (bits[:, :n_q] @ H[:, n_q:].T + bits[:, n_q:] @ H[:, :n_q].T) & 1
        syndrome_idx = 0
        for a in range(n_a):
            # bits[:n_q] @ H[:, n_q:]  -> b_first  & H_Z_ints
            # bits[n_q:] @ H[:, :n_q]  -> b_second & H_X_ints
            overlap = (b_first & H_Z_ints[a]) ^ (b_second & H_X_ints[a])
            if parity(overlap):
                syndrome_idx |= (1 << a)
                
        # --- Logical Observables ---
        # observable_X = np.sum(bits[:,np.where(obs[0])[0]], axis = 1)%2        
        # Extract parity from first half and second half based on obs mask
        obs_0_val = parity(i & mask_obs_0)
        obs_1_val = parity(i & mask_obs_1)
        
        pattern_proba_local[tid, obs_0_val, obs_1_val, syndrome_idx] += prob
        
    pattern_proba = np.zeros((2, 2, 1 << n_a), dtype=np.float64)
    for t in range(num_threads):
        pattern_proba += pattern_proba_local[t]
        
    return pattern_proba

def compute_stab_logical_dist(H, obs, p, eta):
    n_a, n = H.shape
    n_q = n // 2
    
    px = p/2/(1 + eta)
    pz = p*eta/(1 + eta)
    py = p/2/(1 + eta)
    p_I = 1-p
    
    proba_error = np.zeros((n_q + 1, n_q + 1, n_q + 1), dtype=np.float64)
    for nx in range(n_q + 1):
        for ny in range(n_q + 1 - nx):
            for nz in range(n_q + 1 - nx - ny):
                ni = n_q - nx - ny - nz
                proba_error[nx, ny, nz] = (px**nx) * (py**ny) * (pz**nz) * (p_I**ni)
                
    # Compress H
    H_X_ints = np.zeros(n_a, dtype=np.uint64)
    H_Z_ints = np.zeros(n_a, dtype=np.uint64)
    for i in range(n_a):
        hx, hz = 0, 0
        for j in range(n_q):
            if H[i, j]: hx |= (1 << j)
            if H[i, j + n_q]: hz |= (1 << j)
        H_X_ints[i] = hx
        H_Z_ints[i] = hz

    # Compress Obs
    mask_obs_0 = 0
    mask_obs_1 = 0
    for idx in np.where(obs[0])[0]:
        mask_obs_0 |= (1 << int(idx))
    for idx in np.where(obs[1])[0]:
        mask_obs_1 |= (1 << int(idx))
        
    return compute_core(n_a, n_q, H_X_ints, H_Z_ints, proba_error, 1 << n, mask_obs_0, mask_obs_1)


def logical_error(H, obs, p, eta):
    # probability distribution of stabilizers and observable
    stab_logical_dist = compute_stab_logical_dist(H,obs, p, eta)
    # sum least probable element of each class 
    P_L = np.sum(
        np.sum(stab_logical_dist, axis=(0,1)) 
        - np.max(stab_logical_dist, axis=(0,1))
    )
    return P_L


Logical_errors = np.zeros(3**9)

for index, Clifford_deformed in enumerate(product(range(3), repeat=9)):
    print(f"\r{index+1}/{3**9}".ljust(50), end="", flush=True)
    H = Create_parity_check(Clifford_deformed)
    obs = Create_logicals(Clifford_deformed)
    Logical_errors[index] = logical_error(H, obs, p = 0.01, eta = 500)
    
def H2(p):
    return -p*np.log2(p) -(1-p)*np.log2(1-p)
    
def H_L_knowing_S(H, obs, p, eta):
    stab_logical_dist = compute_stab_logical_dist(H,obs, p, eta)
    proba_syndrome = np.sum(stab_logical_dist, axis=(0,1))
    proba_logical_knowing_syndrome = np.divide(
        stab_logical_dist,
        proba_syndrome[None, None, :],
        out=np.zeros_like(stab_logical_dist),
        where=proba_syndrome[None, None, :] > 0,
    )
    nonzero = proba_logical_knowing_syndrome > 0
    entropy = -np.sum(
        stab_logical_dist[nonzero]
        * np.log2(proba_logical_knowing_syndrome[nonzero])
    )
    return entropy
    

H_L_knowing_S_list = np.zeros(3**9)

for index, Clifford_deformed in enumerate(product(range(3), repeat=9)):
    print(f"\r{index+1}/{3**9}".ljust(50), end="", flush=True)
    H = Create_parity_check(Clifford_deformed)
    obs = Create_logicals(Clifford_deformed)
    H_L_knowing_S_list[index] = H_L_knowing_S(H,obs,p = 0.01,eta = 500)    

matplotlib.use('Agg')
plt.style.use('rgplot')
plt.figure(figsize = (2,1.75))
plt.plot(H_L_knowing_S_list,Logical_errors,'o', markersize=2)
plt.xlabel(r'$H(L|S)$', fontsize = 9)
plt.loglog()
plt.ylabel(r'$p_L$', fontsize = 9)

x_limits = plt.xlim()
entropy_bound = np.logspace(np.log10(x_limits[0]), np.log10(x_limits[1]), 500)
plt.plot(
    entropy_bound,
    entropy_bound/2,
    '--',
    c='k',
    linewidth=0.8,
    label=r'$2p_L=H(L|S)$',
)

p_L_bound = np.logspace(-16, np.log10(3/4), 2000)
fano_entropy_bound = H2(p_L_bound) + p_L_bound*np.log2(3)
fano_visible = (
    (fano_entropy_bound >= x_limits[0])
    & (fano_entropy_bound <= x_limits[1])
)
plt.plot(
    fano_entropy_bound[fano_visible],
    p_L_bound[fano_visible],
    ':',
    c='k',
    linewidth=0.8,
    label=r'$H(L|S)=H_2(p_L)+p_L\log_2 3$',
)

plt.xlim(x_limits)
plt.xticks(fontsize = 7)
plt.yticks(fontsize = 7)

plt.savefig('Clifford_deformed.pdf')
