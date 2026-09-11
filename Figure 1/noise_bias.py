import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import fsolve


# error entropy
def H(p,eta):
    pz = p*eta/(1+eta)
    px = p*1/2/(1+eta)
    py = p*1/2/(1+eta)
    return - (1-p)*np.log2(1-p) - px*np.log2(px) - py*np.log2(py) - pz*np.log2(pz)


# binary entropy
def H2(p):
    return - (1-p)*np.log2(1-p) - p*np.log2(p)


def syndrome_proba_xzzx(p, eta):
    pz = p*eta/(1+eta)
    px = p*1/2/(1+eta)
    py = p*1/2/(1+eta)
    
    # anti-commutation properties
    error_properties = {
        'I': (1-p, False, False),
        'X': (px, False, True),
        'Y': (py, True, True),
        'Z': (pz, True, False)
    }
    
    proba = 0
    
    for err1, err2, err3, err4 in product(error_properties.keys(), repeat=4):
        is_negative_syndrome = (
            error_properties[err1][2] ^ 
            error_properties[err2][1] ^ 
            error_properties[err3][1] ^ 
            error_properties[err4][2]
        )
        
        if is_negative_syndrome:
            proba += (
                error_properties[err1][0] 
                * error_properties[err2][0]
                * error_properties[err3][0] 
                * error_properties[err4][0]
            )
            
    return proba


def syndrome_Z_proba_css(p, eta):
    pz = p*eta/(1+eta)
    px = p*1/2/(1+eta)
    py = p*1/2/(1+eta)
    
    # anti-commutation properties
    error_properties = {
        'I': (1-p, False),
        'X': (px, True),
        'Y': (py, True),
        'Z': (pz, False)
    }
    
    proba = 0
    
    for err1, err2, err3, err4 in product(error_properties.keys(), repeat=4):
        is_negative_syndrome = (
            error_properties[err1][1] ^ 
            error_properties[err2][1] ^ 
            error_properties[err3][1] ^ 
            error_properties[err4][1]
        )
        
        if is_negative_syndrome:
            proba += (
                error_properties[err1][0] 
                * error_properties[err2][0]
                * error_properties[err3][0] 
                * error_properties[err4][0]
            )
            
    return proba


def syndrome_X_proba_css(p, eta):
    pz = p*eta/(1+eta)
    px = p*1/2/(1+eta)
    py = p*1/2/(1+eta)
    
    # anti-commutation properties
    error_properties = {
        'I': (1-p, False),
        'X': (px, False),
        'Y': (py, True),
        'Z': (pz, True)
    }
    
    proba = 0
    
    for err1, err2, err3, err4 in product(error_properties.keys(), repeat=4):
        is_negative_syndrome = (
            error_properties[err1][1] ^ 
            error_properties[err2][1] ^ 
            error_properties[err3][1] ^ 
            error_properties[err4][1]
        )
        
        if is_negative_syndrome:
            proba += (
                error_properties[err1][0] 
                * error_properties[err2][0]
                * error_properties[err3][0] 
                * error_properties[err4][0]
            )
            
    return proba
            
            
def syndrome_XY_proba(p, eta):
    pz = p*eta/(1+eta)
    px = p*1/2/(1+eta)
    py = p*1/2/(1+eta)
    
    # anti-commutation properties
    error_properties = {
        'I': (1-p, False),
        'X': (px, True),
        'Y': (py, False),
        'Z': (pz, True)
    }
    
    proba = 0
    
    for err1, err2, err3, err4 in product(error_properties.keys(), repeat=4):
        is_negative_syndrome = (
            error_properties[err1][1] ^ 
            error_properties[err2][1] ^ 
            error_properties[err3][1] ^ 
            error_properties[err4][1]
        )
        
        if is_negative_syndrome:
            proba += (
                error_properties[err1][0] 
                * error_properties[err2][0]
                * error_properties[err3][0] 
                * error_properties[err4][0]
            )    
            
    return proba


def pure_z_negative_syndrome_probability(p, weight):
    return (1 - (1 - 2*p)**weight) / 2


# noise bias
eta_list = np.logspace(np.log10(0.5), 3,100)

threhsold_xzzx = np.zeros(len(eta_list))

for index, eta in enumerate(eta_list):    
    def residual_noise(p):
        return H(p,eta) - H2(syndrome_proba_xzzx(p,eta))

    pth = fsolve(residual_noise, x0 = 0.4)[0]

    threhsold_xzzx[index] = pth
    
    
threhsold_sc = np.zeros(len(eta_list))

for index, eta in enumerate(eta_list):    
    def residual_noise(p):
        return H(p,eta) - 0.5*H2(syndrome_X_proba_css(p,eta)) - 0.5*H2(syndrome_Z_proba_css(p,eta))

    pth = fsolve(residual_noise, x0 = 0.1)[0]

    threhsold_sc[index] = pth
    
    
threhsold_xy = np.zeros(len(eta_list))

for index, eta in enumerate(eta_list):    
    def residual_noise(p):
        return H(p,eta) - H2(syndrome_XY_proba(p,eta))

    pth = fsolve(residual_noise, x0 = 0.4)[0]

    threhsold_xy[index] = pth


def residual_noise_infinity_xzzx(p):
    return H2(p) - H2(pure_z_negative_syndrome_probability(p, 2))


def residual_noise_infinity_css(p):
    return H2(p) - 0.5*H2(pure_z_negative_syndrome_probability(p, 4))


def residual_noise_infinity_xy(p):
    return H2(p) - H2(pure_z_negative_syndrome_probability(p, 4))


threshold_infinity_xzzx = fsolve(residual_noise_infinity_xzzx, x0 = 0.4)[0]
threshold_infinity_css = fsolve(residual_noise_infinity_css, x0 = 0.1)[0]
threshold_infinity_xy = fsolve(residual_noise_infinity_xy, x0 = 0.4)[0]
    
    
noise_bias_mld = [0.5, 1, 3, 10, 30, 100, 300, 1000, np.inf]

# from https://bitbucket.org/qecsim/qsdxzzx/
# from https://arxiv.org/abs/2009.07851
threshold_mld_xzzx = [
    0.18724504285638313,
    0.19208454962483068,
    0.2208214913015228,
    0.28233070415890205,
    0.3475986253197999,
    0.4060240636189221,
    0.45713807144602275,
    0.4932370362359175,
    0.5,
]

# from https://bitbucket.org/qecsim/qsdxzzx/
# from https://arxiv.org/abs/2009.07851
threshold_mld_css = [
    0.18777621207320944,
    0.1770377329150635,
    0.13338052362223493,
    0.11615253437467293,
    0.11071969126210712,
    0.10917351047939305,
    0.10851663800241862,
    0.10906794614374765,
    0.1094
]

# from https://arxiv.org/abs/1812.08186
threshold_mld_xy = [
    0.188,
    0.194,
    0.223,
    0.281,
    0.339,
    0.392,
    0.429,
    0.454,
    0.5,
]

plt.figure()    

plt.style.use('rgplot')

figure, axis = plt.subplots(
    figsize=(2.75,1.75),
)
    
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

axis.plot([],[],'o', c='k', label = 'tensor network decoder')
axis.plot(eta_list,threhsold_sc,'-', label = 'CSS surface code', c = colors[1])
axis.plot(eta_list,threhsold_xy,'-', label = 'XY code', c = colors[2])
axis.plot(eta_list,threhsold_xzzx,'-', label = 'XZZX code', c = colors[0])
axis.plot(noise_bias_mld[:-1],threshold_mld_css[:-1],'o', c = colors[1])
axis.plot(noise_bias_mld[:-1],threshold_mld_xzzx[:-1],'o', c = colors[0])
axis.plot(noise_bias_mld[:-1],threshold_mld_xy[:-1],'o', c = colors[2])
axis.set_xscale('log')

axis.plot([4200],[threshold_infinity_css],'_', markersize=10, c = colors[1], zorder=104)
axis.plot([4200],[threshold_infinity_xy],'_', markersize=10, c = colors[2], zorder=104)
axis.plot([4200],[threshold_infinity_xzzx],'_', markersize=10, c = colors[0], zorder=104)
axis.plot([4200],[threshold_mld_xzzx[-1]],'o', c = colors[0], zorder=104)
axis.plot([4200],[threshold_mld_xy[-1]],'o', c = colors[2], zorder=104)
axis.plot([4200],[threshold_mld_css[-1]],'o', c = colors[1], zorder=104)

axis.set_xlim(0.5, 6000)
axis.set_xticks([1, 10, 100, 1000])
axis.set_xticklabels([
    r'$10^0$',
    r'$10^1$',
    r'$10^2$',
    r'$10^3$',
])
axis.text(
    4200,
    -0.03,
    r'$\infty$',
    transform=axis.get_xaxis_transform(),
    ha='center',
    va='top',
    fontsize=9,
    zorder=104,
)
axis.plot(
    (4200, 4200),
    (0, 0.025),
    transform=axis.get_xaxis_transform(),
    color='k',
    clip_on=False,
    linewidth=1.0,
    zorder=104,
)


break_size = 0.025
break_style = dict(
    color='k',
    clip_on=False,
    linewidth=0.8,
    transform=axis.get_xaxis_transform(),
    zorder=105,
)
for break_position in [1850, 2200]:
    axis.plot(
        (break_position/1.06, break_position*1.06),
        (-break_size, break_size),
        **break_style,
    )
    axis.plot(
        (break_position/1.06, break_position*1.06),
        (1-break_size, 1+break_size),
        **break_style,
    )

axis.tick_params(axis='both', labelsize=7)
axis.tick_params(axis='y', which='both', right=False)
grid_end_fraction = (
    (np.log10(1500) - np.log10(0.5))
    / (np.log10(6000) - np.log10(0.5))
)
for tick in axis.yaxis.get_major_ticks() + axis.yaxis.get_minor_ticks():
    tick.gridline.set_xdata([0, grid_end_fraction])
axis.set_ylabel(r'$p_{th}$', fontsize=9)
axis.set_xlabel(r'$\eta$', fontsize=9)
axis.legend(loc='upper left', fontsize=7)

figure.savefig('noise_bias.pdf')
