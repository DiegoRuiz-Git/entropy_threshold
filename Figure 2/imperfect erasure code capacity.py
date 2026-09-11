import numpy as np
import random
import json
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('Agg')

# determine threshold through crossing
def spline_crossing(Logical_error, Logical_error_u, p_list, n_boot=1000, verbose=False, rng=None):
    rng = np.random.default_rng(rng)
    Logical_error = np.asarray(Logical_error)
    Logical_error_u = np.asarray(Logical_error_u)

    def fit_and_cross(y0, y1):
        s0 = UnivariateSpline(p_list, y0, w=1 / Logical_error_u[0], s=None)
        s1 = UnivariateSpline(p_list, y1, w=1 / Logical_error_u[1], s=None)
        return brentq(lambda x: s0(x) - s1(x), p_list[0], p_list[-1]), s0, s1

    p_cross, s9, s11 = fit_and_cross(Logical_error[0], Logical_error[1])

    crossings = np.empty(n_boot)
    for k in range(n_boot):
        y0 = rng.normal(Logical_error[0], Logical_error_u[0])
        y1 = rng.normal(Logical_error[1], Logical_error_u[1])
        try:
            crossings[k], _, _ = fit_and_cross(y0, y1)
        except ValueError:
            crossings[k] = np.nan

    p_cross_u = np.nanstd(crossings)

    if verbose:
        pp = np.linspace(p_list[0], p_list[-1], 500)
        for i, d in zip([0, 1], [9, 11]):
            plt.plot(p_list, Logical_error[i], 'o', label=f'd = {d}')
            plt.fill_between(p_list,
                            Logical_error[i] - Logical_error_u[i],
                            Logical_error[i] + Logical_error_u[i],
                            alpha=0.4)
        plt.plot(pp, s9(pp), '-')
        plt.plot(pp, s11(pp), '-')
        plt.axvspan(p_cross - p_cross_u, p_cross + p_cross_u,
                    color='k', alpha=0.2)
        plt.axvline(p_cross, ls='--', color='k',
                    label=f'crossing = {p_cross:.4f} ± {p_cross_u:.4f}')
        plt.legend()
        plt.show()

    return p_cross, p_cross_u


# load cluster stim simulation
q_list = np.linspace(0,0.5,10)

threshold_list = np.zeros(len(q_list))
threshold_u_list = np.zeros(len(q_list))

for index_q, q in enumerate(q_list):
    if index_q>=1:
        d_list = np.arange(9,13,2)
    
        logical_error = np.zeros((len(d_list), 20))
        logical_error_u = np.zeros((len(d_list), 20))
    
        for index_d, d in enumerate(d_list):
            for index_p in range(20):      
                with open(f"cluster imperfect erasure code capacity/result/result_{index_q}_{d}_{index_p}.json", "r") as f:
                    data = json.load(f)

                p_list = np.array(data['p_e_list'])
                tot_errors = data["nb_mistakes"]
                tot_shots = data["nb_shots"]
                
                ler = tot_errors/tot_shots
                ler_u = np.sqrt(1/tot_shots*ler*(1-ler))
                logical_error[index_d, index_p] = ler
                logical_error_u[index_d, index_p] = ler_u
                
        p_cross, p_cross_u = spline_crossing(logical_error, logical_error_u, p_list, verbose = False)
        threshold_list[index_q] = p_cross
        threshold_u_list[index_q] = p_cross_u
        
threshold_list[0] = 0.5
threshold_u_list[0] = 0


# binary entropy
def plogp(p):
    return p * np.log2(p) if p > 0 else 0

# error entropy
def H_E_knowing_F(p,qup,qdown,percentage):
    pe = percentage*p
    pp = (1-percentage)*p
    
    proba_F0 = (1-pe)*(1-qup) + pe*qdown
    proba_F1 = pe*(1-qdown) + (1-pe)*qup
    
    return (H_E_knowing_F0(pp,pe,qup,qdown)*proba_F0 
            + H_E_knowing_F1(pp,pe,qup,qdown)*proba_F1)


# error entropy at fixed erasure flag value F=0
def H_E_knowing_F0(pp,pe,qup,qdown):
    if (pe == 1 and qdown == 0) or (pe == 0 and qup == 1):
        proba_erasure_knowing_F0 = 1
    else:
        proba_erasure_knowing_F0 = qdown*pe/((1-pe)*(1-qup)+pe*qdown)
    
    if (pe == 1 and qdown == 0) or (pe == 0 and qup == 1):
        proba_no_erasure_knowing_F0 = 1
    else:
        proba_no_erasure_knowing_F0 = (1-pe)*(1-qup)/((1-pe)*(1-qup)+pe*qdown)
    
    proba_I_knowing_F0 = (1/4*proba_erasure_knowing_F0 
                          + (1-pp)*proba_no_erasure_knowing_F0)
    proba_X_knowing_F0 = (1/4*proba_erasure_knowing_F0 
                          + pp/3*proba_no_erasure_knowing_F0)
    proba_Y_knowing_F0 = proba_X_knowing_F0
    proba_Z_knowing_F0 = proba_X_knowing_F0

    return -(
        plogp(proba_I_knowing_F0) +
        plogp(proba_X_knowing_F0) +
        plogp(proba_Y_knowing_F0) +
        plogp(proba_Z_knowing_F0)
    )


# error entropy at fixed erasure flag value F=1
def H_E_knowing_F1(pp,pe,qup,qdown):
    if (qup == 0 and pe == 0) or (qdown == 1 and pe == 1):
        proba_erasure_knowing_F1 = 1
    else:
        proba_erasure_knowing_F1 = (1-qdown)*pe/((1-pe)*qup+pe*(1-qdown))

    if (qup == 0 and pe == 0) or (qdown == 1 and pe == 1):
        proba_no_erasure_knowing_F1 = 0
    else:
        proba_no_erasure_knowing_F1 = qup*(1-pe)/((1-pe)*qup+pe*(1-qdown))

    proba_I_knowing_F1 = (1/4*proba_erasure_knowing_F1 
                          + (1-pp)*proba_no_erasure_knowing_F1)
    proba_X_knowing_F1 = (1/4*proba_erasure_knowing_F1 
                          + pp/3*proba_no_erasure_knowing_F1)
    proba_Y_knowing_F1 = proba_X_knowing_F1
    proba_Z_knowing_F1 = proba_X_knowing_F1
    
    return -(
        plogp(proba_I_knowing_F1) +
        plogp(proba_X_knowing_F1) +
        plogp(proba_Y_knowing_F1) +
        plogp(proba_Z_knowing_F1)
    )



# stabilizer entropy
def H_S_knowing_F(p,qup,qdown,percentage):
    pe = percentage*p
    pp = (1-percentage)*p

    proba_F0 = (1-pe)*(1-qup) + pe*qdown
    proba_F1 = pe*(1-qdown) + (1-pe)*qup

    if proba_F0 == 0:
        proba_no_erasure_knowing_F0 = 0
    else:
        proba_no_erasure_knowing_F0 = (1-pe)*(1-qup)/proba_F0

    if proba_F1 == 0:
        proba_no_erasure_knowing_F1 = 0
    else:
        proba_no_erasure_knowing_F1 = (1-pe)*qup/proba_F1

    proba_S_pauli = syndrome_proba(pp)
    H_S = 0

    for i, j, k, l in np.ndindex(2, 2, 2, 2):
        hamming_weight = i+j+k+l

        proba_Fpattern = (
            proba_F1**hamming_weight
            *proba_F0**(4-hamming_weight)
        )
        proba_no_erasure_knowing_Fpattern = (
            proba_no_erasure_knowing_F1**hamming_weight
            *proba_no_erasure_knowing_F0**(4-hamming_weight)
        )
        proba_S_knowing_Fpattern = (
            syndrome_proba(pp)*proba_no_erasure_knowing_Fpattern
            + 1/2*(1-proba_no_erasure_knowing_Fpattern)
        )

        H_S += (
            -plogp(proba_S_knowing_Fpattern)
            -plogp(1-proba_S_knowing_Fpattern)
        )*proba_Fpattern

    return H_S


# proba of syndrome being -1 under Pauli noise
def syndrome_proba(p):
    pz = p/3
    px = p/3
    py = p/3

    possible_error = ['I','X','Y','Z']
    proba_error = [1-px-py-pz, px, py, pz]
    proba = 0

    for index1, qubit1 in enumerate(possible_error):
        for index2, qubit2 in enumerate(possible_error):
            for index3, qubit3 in enumerate(possible_error):
                for index4, qubit4 in enumerate(possible_error):
                    # value stab
                    stab_value = 0
                    if qubit1 == 'X' or qubit1 == 'Y':
                        stab_value = (stab_value+1)%2
                    if qubit2 == 'X' or qubit2 == 'Y':
                        stab_value = (stab_value+1)%2
                    if qubit3 == 'X' or qubit3 == 'Y':
                        stab_value = (stab_value+1)%2
                    if qubit4 == 'X' or qubit4 == 'Y':
                        stab_value = (stab_value+1)%2
                    # proba
                    if stab_value == 1:
                        proba += proba_error[index1]*proba_error[index2]*proba_error[index3]*proba_error[index4]

    return proba


# compute thresholds
percentage = 1
q_list_entropy = np.linspace(0,0.5,100)

threhsold_erasure = np.zeros(len(q_list_entropy))

for index, q in enumerate(q_list_entropy):
    def residual_noise(p):
        return H_E_knowing_F(p,qup = q, qdown = q,percentage = percentage) - H_S_knowing_F(p,qup = q, qdown = q,percentage = percentage)

    pc = fsolve(residual_noise, x0 = 0.4)[0]

    threhsold_erasure[index] = pc
    
    
plt.style.use('rgplot')
plt.figure(figsize = (3,2))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.plot([],[],'o',label='MLE decoder', c = colors[0])
plt.errorbar(q_list, threshold_list, yerr= threshold_u_list, fmt='o', capsize=5, c = colors[0])
plt.plot(q_list_entropy, threhsold_erasure,'-',label = 'entropy', c = colors[0])
plt.ylabel('threshold')
plt.legend(fontsize = 7)
plt.ylabel(r'$p_{th}$', fontsize = 9)
plt.xlabel(r'$q$', fontsize = 9)
plt.xticks(fontsize = 7)
plt.yticks(fontsize = 7)
plt.savefig('imperfect erasure capacity.pdf')

    
