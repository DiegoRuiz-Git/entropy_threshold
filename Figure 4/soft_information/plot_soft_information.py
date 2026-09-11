#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad, quad
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib

matplotlib.use('Agg')

plt.style.use("rgplot")

distances = (17, 19)
hard_threshold = 0.033
p_list = np.linspace(0.02, 0.1, 20)


def load_cluster_results():
    records = {}

    for path in sorted(Path("cluster/result").glob("*.json")):
        with path.open() as file:
            data = json.load(file)

        d = int(data["d"])

        p_index = int(data["p_index"])
        p = float(data["p"])
        shots = int(data["shots"])
        failures = int(data["failures"])
        records[d, p_index] = (p, failures / shots)

    return records


def estimate_crossing(records):
    splines = {}
    p_values = {}

    for d in distances:
        rows = sorted(
            (p, rate)
            for (distance, _), (p, rate) in records.items()
            if distance == d
        )
        p_values[d] = np.array([p for p, _ in rows])
        rates = np.array([rate for _, rate in rows])
        splines[d] = UnivariateSpline(p_values[d], rates, k=3, s=0)

    p_min = max(p_values[d].min() for d in distances)
    p_max = min(p_values[d].max() for d in distances)
    grid = np.linspace(p_min, p_max, 10_000)
    difference = splines[distances[0]](grid) - splines[distances[1]](grid)
    crossing_index = np.where(difference[:-1] * difference[1:] < 0)[0][0]

    return brentq(
        lambda p: splines[distances[0]](p) - splines[distances[1]](p),
        grid[crossing_index],
        grid[crossing_index + 1],
    )


def H2(p):
    if p <= 0 or p>=1:
        return 0
    return - p * np.log2(p) - (1-p) * np.log2(1-p)


def H_E(p):
    return 3*H2(p)


def H_S(p):
    prob_product = ((1 - 2*p)**4) * ((1 - 2*p)**2)
    return H2(0.5 * (1 - prob_product))


h_s_hard_function = H_S


def H_S(p):
    # Two adjacent detectors share each analog measurement. Therefore,
    # the entropy density contains one I(M; Q) per detector, not two.
    measurement_information = 0.5 * (
        h_Q1_Q2(p) - h_Q1_Q2_knowing_E(p)
    )
    return H_D_knowing_Q1_Q2(p) + measurement_information


def h_Q1_Q2(p):

    sigma = 0.5/norm.ppf(1 - p)

    def f(q):
        prefactor = sigma / np.sqrt(2*np.pi) * 1/(q*(1-q))
        l_minus = - (sigma**2*np.log((1-q)/q) - 0.5)**2/2/sigma**2
        l_plus = - (sigma**2*np.log((1-q)/q) + 0.5)**2/2/sigma**2
        return prefactor * (np.exp(l_minus) + np.exp(l_plus))

    res, err = quad(lambda q: -2* f(q) * np.log2(f(q)), 0, 0.5)

    return res


def H_D_knowing_Q1_Q2(p):

    sigma = 0.5/norm.ppf(1 - p)

    def f(q):
        prefactor = sigma / np.sqrt(2*np.pi) * 1/(q*(1-q))
        l_minus = - (sigma**2*np.log((1-q)/q) - 0.5)**2/2/sigma**2
        l_plus = - (sigma**2*np.log((1-q)/q) + 0.5)**2/2/sigma**2
        return prefactor * (np.exp(l_minus) + np.exp(l_plus))

    def integrand(q1,q2):
        return f(q1)*f(q2)*H2(0.5*(1 - (1-2*p)**4*(1-2*q1)*(1-2*q2)))

    res, err = dblquad(integrand,0, 0.5,lambda x: 0,lambda x: 0.5)

    return res


def h_Q1_Q2_knowing_E(p):
    return 2*(p*h_Q1_knowing_error(p) + (1-p)*h_Q1_knowing_no_error(p))


def h_Q1_knowing_no_error(p):

    def f(q):
        sigma = 0.5/norm.ppf(1 - p)
        prefactor = sigma/np.sqrt(2*np.pi)*1/(q*(1-q))/(1-p)
        l_value = - (sigma**2*np.log((1-q)/q) - 0.5)**2/2/sigma**2
        return prefactor * np.exp(l_value)

    res, err = quad(lambda q: - f(q) * np.log2(f(q)), 0, 0.5)

    return res


def h_Q1_knowing_error(p):

    def f(q):
        sigma = 0.5/norm.ppf(1 - p)
        prefactor = sigma/np.sqrt(2*np.pi)*1/(q*(1-q))/p
        l_value = - (sigma**2*np.log((1-q)/q) + 0.5)**2/2/sigma**2
        return prefactor * np.exp(l_value)

    res, err = quad(lambda q: - f(q) * np.log2(f(q)), 0, 0.5)

    return res


h_s_soft_function = H_S


def main():
    records = load_cluster_results()
    soft_threshold = estimate_crossing(records)

    h_e = np.array([H_E(p) for p in p_list])
    h_s_hard = np.array([h_s_hard_function(p) for p in p_list])
    h_s_soft = []
    for index, p in enumerate(p_list, start=1):
        print(
            f"Computing soft entropy {index}/{len(p_list)} (p={p:.5f})",
            flush=True,
        )
        h_s_soft.append(h_s_soft_function(p))
    h_s_soft = np.array(h_s_soft)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.plot(
        p_list,
        h_e,
        color="C1",
        label=r"$H(E)$",
    )
    ax.plot(
        p_list,
        h_s_soft,
        color="C2",
        label=r"$I(E;S)_{\mathrm{soft}}$",
    )
    ax.plot(
        p_list,
        h_s_hard,
        color="C0",
        label=r"$H(S)_{\mathrm{hard}}$",
    )
    ax.axvline(
        soft_threshold,
        color="black",
        linestyle=(0, (3, 1)),
    )
    ax.axvline(
        hard_threshold,
        color="black",
        linestyle=(0, (3, 1)),
    )
    
    ax.set_xlim((p_list[0], p_list[-1]))
    ax.loglog()

    ax.set_xlabel(r"$p$", fontsize=9)
    ax.set_ylabel("Entropy", fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.tick_params(axis="both", which="minor", labelbottom=False, labelleft=False)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("soft_information.pdf", bbox_inches="tight")

    print(
        f"Soft crossing d={distances[0]}/{distances[1]}: "
        f"p = {soft_threshold:.8f}"
    )


if __name__ == "__main__":
    main()
