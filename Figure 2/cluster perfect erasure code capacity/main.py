from gurobipy import Env, Model, GRB
import stim
import numpy as np
import sys
import os
from scipy.optimize import fsolve
import time
import json
import contextlib, io
from stimbposd import BPOSD
import gurobipy as gp
from multiprocessing import Pool
from functools import partial

ISV_CREDS = ******

# code parameter
percent_index = int(sys.argv[1])
d = int(sys.argv[2])
p_index = int(sys.argv[3])

### Create Circuit
def Create_Ancilla_type(d):
    Ancilla_type = np.full((d+1,d+1),' ')

    for i in range(d+1):
        for j in range(d+1):
            if (j+i)%2 == 1 and j>0 and j<d:
                Ancilla_type[i,j] = 'Z'
            if (j+i)%2 == 0 and i>0 and i<d:
                Ancilla_type[i,j] = 'X'

    return Ancilla_type


def Create_X_stabilizers(d):

    Stabilizers_X = np.full((int((d**2-1)/2),4),np.nan)

    Ancilla_type = Create_Ancilla_type(d)
    Data_number = np.array(range(d**2)).reshape((d,d))

    counter = 0

    for i in range(Ancilla_type.shape[0]):
        for j in range(Ancilla_type.shape[1]):

            if Ancilla_type[i,j] == "X":
                if i>0 and j>0:
                    Stabilizers_X[counter][0] = Data_number[i-1,j-1]
                if i<d and j>0:
                    Stabilizers_X[counter][1] = Data_number[i,j-1]
                if i>0 and j<d:
                    Stabilizers_X[counter][2] = Data_number[i-1,j]
                if i<d and j<d:
                    Stabilizers_X[counter][3] = Data_number[i,j]
                counter += 1

    return Stabilizers_X


def Create_Z_stabilizers(d):

    Stabilizers_Z = np.full((int((d**2-1)/2),4),np.nan)

    Ancilla_type = Create_Ancilla_type(d)
    Data_number = np.array(range(d**2)).reshape((d,d))

    counter = 0

    for i in range(Ancilla_type.shape[0]):
        for j in range(Ancilla_type.shape[1]):

            if Ancilla_type[i,j] == "Z":
                if i>0 and j>0:
                    Stabilizers_Z[counter][0] = Data_number[i-1,j-1]
                if i>0 and j<d:
                    Stabilizers_Z[counter][1] = Data_number[i-1,j]
                if i<d and j>0:
                    Stabilizers_Z[counter][2] = Data_number[i,j-1]
                if i<d and j<d:
                    Stabilizers_Z[counter][3] = Data_number[i,j]
                counter += 1

    return Stabilizers_Z


def Create_Circuit_erasure(d, p, percent, basis):

    p_pauli = p*(1-percent)

    circuit = stim.Circuit()

    if basis == 'Z':
        circuit.append('R', range(d**2))
    if basis == 'X':
        circuit.append('RX', range(d**2))

    # ----------------- #
    # First round stab
    # ----------------- #

    X_stab = Create_X_stabilizers(d)

    for x_stab in X_stab:

        valid_qubits = [int(qubit) for qubit in x_stab if not np.isnan(qubit)]

        targets = []
        for index, qubit in enumerate(valid_qubits):
            targets.append(stim.target_x(qubit))
            if index < len(valid_qubits) - 1:
                targets.append(stim.target_combiner())

        if targets:
            circuit.append("MPP", targets)

    Z_stab = Create_Z_stabilizers(d)

    for z_stab in Z_stab:

        valid_qubits = [int(qubit) for qubit in z_stab if not np.isnan(qubit)]

        targets = []
        for index, qubit in enumerate(valid_qubits):
            targets.append(stim.target_z(qubit))
            if index < len(valid_qubits) - 1:
                targets.append(stim.target_combiner())

        if targets:
            circuit.append("MPP", targets)

    # ----------------- #
    # Noise
    # ----------------- #

    circuit.append('DEPOLARIZE1', range(d**2), p_pauli)

    # ----------------- #
    # Second round stab
    # ----------------- #

    for x_stab in X_stab:

        valid_qubits = [int(qubit) for qubit in x_stab if not np.isnan(qubit)]

        targets = []
        for index, qubit in enumerate(valid_qubits):
            targets.append(stim.target_x(qubit))
            if index < len(valid_qubits) - 1:
                targets.append(stim.target_combiner())

        if targets:
            circuit.append("MPP", targets)

    for z_stab in Z_stab:

        valid_qubits = [int(qubit) for qubit in z_stab if not np.isnan(qubit)]

        targets = []
        for index, qubit in enumerate(valid_qubits):
            targets.append(stim.target_z(qubit))
            if index < len(valid_qubits) - 1:
                targets.append(stim.target_combiner())

        if targets:
            circuit.append("MPP", targets)

    for i in range(d**2-1):
        circuit.append('DETECTOR',  [stim.target_rec(-1-i), stim.target_rec(-1-i-(d**2-1))])

    # ----------------- #
    # Observable
    # ----------------- #

    if basis == 'Z':
        targets = []
        for index in range(d):
            targets.append(stim.target_z(index*d))
            if index < d-1:
                targets.append(stim.target_combiner())

        circuit.append('MPP', targets)
        circuit.append('OBSERVABLE_INCLUDE', stim.target_rec(-1),0)

    if basis == 'X':
        targets = []
        for index in range(d):
            targets.append(stim.target_x(index))
            if index < d-1:
                targets.append(stim.target_combiner())

        circuit.append('MPP', targets)
        circuit.append('OBSERVABLE_INCLUDE', stim.target_rec(-1),0)

    return circuit


def _decode_chunk(chunk_idx, prefix, suffix, p_e, num_shots, d):
    rng = np.random.default_rng(seed=chunk_idx)

    # Setup Gurobi env (once per chunk)
    with contextlib.redirect_stdout(io.StringIO()):
        env = gp.Env("", True)
        env.setParam("OutputFlag", 0)
        for k_, v in ISV_CREDS.items():
            env.setParam(k_, v)
        env.start()

    # Build a reference circuit (no erasure) to get H, obs structure
    base_circuit = stim.Circuit(prefix + "\n" + suffix)
    base_dem = base_circuit.detector_error_model(approximate_disjoint_errors=True, decompose_errors=False)
    base_decoder = BPOSD(base_dem)
    H = np.array(base_decoder._matrices.check_matrix, dtype=int)
    obs = np.array(base_decoder._matrices.observables_matrix, dtype=int).reshape(-1)
    num_stabilizers, n = H.shape

    # Build Gurobi model ONCE
    model = gp.Model("MLD_Decoder", env=env)
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 1)

    e = model.addVars(n, vtype=GRB.BINARY, name="e")
    k = model.addVars(num_stabilizers, vtype=GRB.INTEGER, name="k")

    # Constraints with placeholder objective and RHS
    constraints = []
    for i in range(num_stabilizers):
        c = model.addConstr(
            gp.quicksum(int(H[i, j]) * e[j] for j in range(n)) - 2 * k[i] == 0,
            name=f"stab_{i}"
        )
        constraints.append(c)

    model.setObjective(gp.quicksum(0.0 * e[j] for j in range(n)), GRB.MINIMIZE)
    model.update()

    nb_errors = 0

    for shot_idx in range(num_shots):
        # Generate erasure pattern
        erased_qubits = np.where(rng.random(size=d**2) < p_e)[0]
        if erased_qubits.size > 0:
            erasure_line = "DEPOLARIZE1(0.75) " + " ".join(map(str, erased_qubits.tolist()))
            circuit_text = prefix + "\n" + erasure_line + "\n" + suffix
        else:
            circuit_text = prefix + "\n" + suffix
        circuit = stim.Circuit(circuit_text)

        # Sample syndrome
        sampler = circuit.compile_detector_sampler(seed=int(rng.integers(0, 2**31 - 1)))
        syndrome, actual_observable = sampler.sample(shots=1, separate_observables=True)

        # Get updated priors (only thing that changes)
        dem = circuit.detector_error_model(approximate_disjoint_errors=True, decompose_errors=False)
        decoder = BPOSD(dem)
        priors = np.asarray(decoder._matrices.priors).reshape(-1)
        priors = np.clip(priors, 1e-12, 1 - 1e-12)
        priors = np.where(priors == 0.5, 0.5 - 1e-9, priors)
        llr_weights = np.log((1.0 - priors) / priors)

        # Update objective coefficients
        for j in range(n):
            e[j].Obj = float(llr_weights[j])

        # Update RHS with syndrome
        s = np.asarray(syndrome[0]).reshape(-1)
        for i in range(num_stabilizers):
            constraints[i].RHS = int(s[i])

        model.optimize()

        if model.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Gurobi failed shot {shot_idx}. Status: {model.status}")

        flip = int(sum(int(round(e[j].X)) * int(obs[j]) for j in range(n)) % 2)
        nb_errors += int(flip != int(actual_observable[0, 0]))

    model.dispose()
    env.dispose()
    return nb_errors

def gurobi_mld_decoder(prefix, suffix, p_e, d, num_workers, batch_size):
    num_chunks = num_workers * 10
    num_shots = batch_size // num_chunks
    actual_total_shots = num_chunks * num_shots

    worker = partial(_decode_chunk, prefix=prefix, suffix=suffix, p_e=p_e, num_shots=num_shots, d=d)

    print(f"Decoding {num_chunks * num_shots} shots in {num_chunks} chunks ({num_shots} shots/chunk)", flush=True)
    start_time = time.time()
    last_print = 0
    interval = 30  # print at most every 30 seconds

    total_errors = 0
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap(worker, range(num_chunks)), start=1):
            total_errors += result
            elapsed = time.time() - start_time
            if elapsed - last_print >= interval or i == num_chunks:
                eta = elapsed / i * (num_chunks - i)
                print(
                    f"Progress: {i}/{num_chunks} chunks ({100*i/num_chunks:.1f}%) | "
                    f"elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
                    flush=True
                )
                last_print = elapsed

    return total_errors, actual_total_shots


if __name__ == "__main__":
    percent_list = np.linspace(0, 1, 10)
    percent = percent_list[percent_index]

    def H_E_knowing_F(p, percentage):
        pe = percentage * p
        pp = (1 - percentage) * p
        if percentage == 1:
            return 2 * pe
        return pe * 2 + (1 - pe) * (-(1 - pp) * np.log2(1 - pp) - pp * np.log2(pp / 3))

    def residual_noise(p):
        return H_E_knowing_F(p, percentage=percent) - 1

    hashing_bound = fsolve(residual_noise, x0=0.4)[0]

    low = max(hashing_bound - 0.05, 1e-4)
    high = min(hashing_bound + 0.05, 0.5)
    p_list = np.logspace(np.log10(low), np.log10(high), 20)
    p = p_list[p_index]
    p_e = p * percent

    print(f"percent={percent}, p={p:.4f}, p_e={p_e:.4f}", flush=True)

    batch_size = 100_000

    circuit = Create_Circuit_erasure(d, p, percent, basis='Z')
    circuit_lines = str(circuit).splitlines()
    prefix = "\n".join(circuit_lines[:2])
    suffix = "\n".join(circuit_lines[2:])

    num_workers = os.cpu_count()
    print(f"Using {num_workers} workers", flush=True)

    start = time.time()
    nb_errors, actual_total_shots = gurobi_mld_decoder(prefix, suffix, p_e, d, num_workers, batch_size)
    end = time.time()

    print(f"Execution time: {end - start:.1f} seconds")
    print(f"Errors: {nb_errors}/{actual_total_shots}")

    result_dict = {
        'd': d,
        'p': float(p),
        'percent': float(percent),
        'percent_index': percent_index,
        'p_index': p_index,
        'p_list': p_list.tolist(),
        'nb_shots': actual_total_shots,
        'nb_mistakes': int(nb_errors),
    }

    with open(f"result/result_{percent_index}_{d}_{p_index}.json", "w") as file:
        json.dump(result_dict, file)
