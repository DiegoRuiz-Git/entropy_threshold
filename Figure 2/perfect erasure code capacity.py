import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('Agg')

### entropy

# error entropy
def H_E_knowing_F(p,percentage):
    pe = percentage*p
    pp = (1-percentage)*p
    if percentage == 1:
        return 2*pe
    return pe*(2) + (1-pe)*(- (1-pp)*np.log2(1-pp) - pp*np.log2(pp/3))

# syndrome entropy
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

def H2(p):
    if p==0:
        return - (1-p)*np.log2(1-p)
    return - (1-p)*np.log2(1-p) - p*np.log2(p)

def H_S_knowing_F(p,percentage):
    pe = percentage*p
    pp = (1-percentage)*p
    p_no_erasure = (1 - pe)**4
    return p_no_erasure*H2(syndrome_proba(pp)) + (1-p_no_erasure)*H2(1/2)

# threshold curve
percentage_list = np.linspace(0,1,100) # percentage of erasure

threhsold_erasure = np.zeros(len(percentage_list))

for index, percentage in enumerate(percentage_list):    
    def residual_noise(p):
        return H_E_knowing_F(p,percentage) - H_S_knowing_F(p,percentage)

    pc = fsolve(residual_noise, x0 = 0.4)[0]

    threhsold_erasure[index] = pc

### simu with gurobi

# determine logical error crossing
def spline_crossing(Logical_error, Logical_error_u, p_list, n_boot=1000, rng=None):
    rng = np.random.default_rng(rng)

    def fit_and_cross(y0, y1):
        s0 = UnivariateSpline(p_list, y0, w=1 / Logical_error_u[0], s=None)
        s1 = UnivariateSpline(p_list, y1, w=1 / Logical_error_u[1], s=None)
        return brentq(lambda x: s0(x) - s1(x), p_list[0], p_list[-1]), s0, s1

    p_cross, s9, s11 = fit_and_cross(Logical_error[0], Logical_error[1])

    # determine uncertainty with Montecarlo
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

# load data
d_list = np.arange(9, 13, 2)

erasure_list = np.linspace(0,1,10)
threshold_list = np.full(len(erasure_list), 0.5)
threshold_u_list = np.zeros(len(erasure_list))

for index_e in range(len(erasure_list[:-1])):
    Logical_error = np.zeros((len(d_list), 20))
    Logical_error_u = np.zeros_like(Logical_error)
    for index_d, d in enumerate(d_list):
        for index_p in range(20):
            with open(f"cluster perfect erasure code capacity/result/result_{index_e}_{d}_{index_p}.json") as f:
                data = json.load(f)
            p_list = data['p_list']
            ler = data['nb_mistakes'] / data['nb_shots']
            Logical_error[index_d, index_p] = ler
            Logical_error_u[index_d, index_p] = np.sqrt(ler * (1 - ler) / data['nb_shots'])
    
    p_cross, p_cross_u = spline_crossing(Logical_error, Logical_error_u, p_list)
    threshold_list[index_e] = p_cross
    threshold_u_list[index_e] = p_cross_u

### plot figure
plt.style.use('rgplot')
plt.figure(figsize = (3,2))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.plot([],[],'o',label='MLE decoder', c = colors[0])
plt.plot(percentage_list, threhsold_erasure,'-',label = 'entropy', c = colors[0])
plt.errorbar(erasure_list,threshold_list,yerr=threshold_u_list,fmt='o',capsize=5, c = colors[0])
plt.legend(fontsize = 7)
plt.ylabel(r'$p_{th}$', fontsize = 9)
plt.xlabel(r'$p_e/p$', fontsize = 9)
plt.xticks(fontsize = 7)
plt.yticks(fontsize = 7)
plt.savefig('perfect erasure capacity.pdf')