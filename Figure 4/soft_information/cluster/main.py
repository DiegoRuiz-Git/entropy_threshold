#!/usr/bin/env python3

import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pymatching
import stim
from scipy.special import expit
from scipy.stats import norm


p_list = np.linspace(0.03, 0.05, 10)
shots = 10**6
seed = 0


def create_layout(d):
    ancilla_type = np.full((d + 1, d + 1), " ")

    for i in range(d + 1):
        for j in range(d + 1):
            if (i + j) % 2 == 1 and 0 < j < d:
                ancilla_type[i, j] = "Z"
            if (i + j) % 2 == 0 and 0 < i < d:
                ancilla_type[i, j] = "X"

    ancilla_number = np.full((d + 1, d + 1), -1, dtype=int)
    data_number = np.zeros((d, d), dtype=int)
    qubit = 0

    for i in range(d + 1):
        for j in range(d + 1):
            if ancilla_type[i, j] != " ":
                ancilla_number[i, j] = qubit
                qubit += 1
        if i < d:
            for j in range(d):
                data_number[i, j] = qubit
                qubit += 1

    ancilla_counter = np.full((d + 1, d + 1), -1, dtype=int)
    counter = 0
    for i in range(d + 1):
        for j in range(d + 1):
            if ancilla_type[i, j] != " ":
                ancilla_counter[i, j] = counter
                counter += 1

    data_counter = np.arange(d * d).reshape(d, d)

    return (
        ancilla_type,
        ancilla_number,
        data_number,
        ancilla_counter,
        data_counter,
    )


def create_circuit(d, p, q):
    (
        ancilla_type,
        ancilla_number,
        data_number,
        ancilla_counter,
        data_counter,
    ) = create_layout(d)

    circuit = stim.Circuit()
    ancilla_measurements = d * d - 1

    for round_index in range(d):
        if round_index == 0:
            circuit.append("R", data_number.flatten())

        for i in range(d + 1):
            for j in range(d + 1):
                if ancilla_type[i, j] == "Z":
                    circuit.append("R", ancilla_number[i, j])
                elif ancilla_type[i, j] == "X":
                    circuit.append("RX", ancilla_number[i, j])

        circuit.append("X_ERROR", data_number.flatten(), p)
        circuit.append("TICK")

        for i in range(1, d + 1):
            for j in range(1, d + 1):
                if ancilla_type[i, j] == "Z":
                    circuit.append(
                        "CX",
                        [data_number[i - 1, j - 1], ancilla_number[i, j]],
                    )
                elif ancilla_type[i, j] == "X":
                    circuit.append(
                        "CX",
                        [ancilla_number[i, j], data_number[i - 1, j - 1]],
                    )
        circuit.append("TICK")

        for i in range(1, d + 1):
            for j in range(1, d + 1):
                if ancilla_type[i, j] == "Z":
                    circuit.append(
                        "CX",
                        [data_number[i - 1, j], ancilla_number[i, j]],
                    )
                elif ancilla_type[i, j] == "X":
                    circuit.append(
                        "CX",
                        [ancilla_number[i, j], data_number[i, j - 1]],
                    )
        circuit.append("TICK")

        for i in range(d):
            for j in range(d):
                if ancilla_type[i, j] == "Z":
                    circuit.append(
                        "CX",
                        [data_number[i, j - 1], ancilla_number[i, j]],
                    )
                elif ancilla_type[i, j] == "X":
                    circuit.append(
                        "CX",
                        [ancilla_number[i, j], data_number[i - 1, j]],
                    )
        circuit.append("TICK")

        for i in range(d):
            for j in range(d):
                if ancilla_type[i, j] == "Z":
                    circuit.append(
                        "CX",
                        [data_number[i, j], ancilla_number[i, j]],
                    )
                elif ancilla_type[i, j] == "X":
                    circuit.append(
                        "CX",
                        [ancilla_number[i, j], data_number[i, j]],
                    )
        circuit.append("TICK")

        for i in range(d + 1):
            for j in range(d + 1):
                if ancilla_type[i, j] == "Z":
                    circuit.append("MZ", ancilla_number[i, j], q)
                elif ancilla_type[i, j] == "X":
                    circuit.append("MX", ancilla_number[i, j], q)

        if round_index == 0:
            for i in range(d + 1):
                for j in range(d + 1):
                    if ancilla_type[i, j] == "Z":
                        circuit.append(
                            "DETECTOR",
                            [
                                stim.target_rec(
                                    -ancilla_measurements
                                    + ancilla_counter[i, j]
                                )
                            ],
                        )
        else:
            for i in range(ancilla_measurements):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-2 * ancilla_measurements + i),
                        stim.target_rec(-ancilla_measurements + i),
                    ],
                )

        if round_index == d - 1:
            circuit.append("M", data_number.flatten(), q)

            for i in range(d + 1):
                for j in range(d + 1):
                    if ancilla_type[i, j] != "Z":
                        continue

                    stabilizer_data = [
                        data_counter[x, y]
                        for x in (i - 1, i)
                        for y in (j - 1, j)
                        if 0 <= x < d and 0 <= y < d
                    ]
                    circuit.append(
                        "DETECTOR",
                        [stim.target_rec(-d * d + x) for x in stabilizer_data]
                        + [
                            stim.target_rec(
                                -2 * d * d + 1 + ancilla_counter[i, j]
                            )
                        ],
                    )

            circuit.append(
                "OBSERVABLE_INCLUDE",
                [
                    stim.target_rec(-d * d + data_counter[i, 0])
                    for i in range(d)
                ],
                0,
            )
        else:
            circuit.append("TICK")

    return circuit


def create_soft_circuit(circuit, q_values):
    soft_circuit = stim.Circuit()
    index = 0

    for instruction in circuit:
        if instruction.name in {"M", "MZ", "MX"}:
            for target in instruction.targets_copy():
                soft_circuit.append(
                    instruction.name,
                    target,
                    float(q_values[index]),
                )
                index += 1
        else:
            soft_circuit.append(instruction)

    return soft_circuit


def simulate_chunk(chunk, d, p, seed_base):
    chunk_index, chunk_shots = chunk
    chunk_seed = (seed_base + chunk_index + 1) & 0xFFFFFFFF

    circuit = create_circuit(d, p, 0)
    sampler = circuit.compile_sampler(seed=chunk_seed)
    converter = circuit.compile_m2d_converter()

    measurements = sampler.sample(chunk_shots).astype(float)
    sigma = 0.5 / norm.ppf(1 - p)
    rng = np.random.default_rng(chunk_seed)
    analog_measurements = measurements + rng.normal(
        0,
        sigma,
        measurements.shape,
    )

    q_values = expit(-np.abs(analog_measurements - 0.5) / sigma**2)
    hard_measurements = analog_measurements >= 0.5
    detection_events, observable_flips = converter.convert(
        measurements=hard_measurements,
        separate_observables=True,
    )

    failures = 0
    for i in range(chunk_shots):
        soft_circuit = create_soft_circuit(circuit, q_values[i])
        dem = soft_circuit.detector_error_model()
        matcher = pymatching.Matching.from_detector_error_model(dem)
        prediction = matcher.decode(detection_events[i])
        failures += int(np.any(prediction != observable_flips[i]))

    return failures


def run_simulation(d, p, num_workers, total_shots, seed_base):
    num_chunks = min(total_shots, max(1, num_workers * 10))
    shots_per_chunk, remainder = divmod(total_shots, num_chunks)
    chunks = [
        (chunk_index, shots_per_chunk + int(chunk_index < remainder))
        for chunk_index in range(num_chunks)
    ]

    worker = partial(
        simulate_chunk,
        d=d,
        p=p,
        seed_base=seed_base,
    )

    print(
        f"Decoding {total_shots} shots in {num_chunks} chunks "
        f"with {num_workers} workers",
        flush=True,
    )

    start_time = time.time()
    last_progress_time = 0
    failures = 0

    with Pool(num_workers) as pool:
        for completed, chunk_failures in enumerate(
            pool.imap_unordered(worker, chunks),
            start=1,
        ):
            failures += chunk_failures
            elapsed = time.time() - start_time

            if elapsed - last_progress_time >= 30 or completed == num_chunks:
                eta = elapsed / completed * (num_chunks - completed)
                print(
                    f"Progress: {completed}/{num_chunks} chunks "
                    f"({100 * completed / num_chunks:.1f}%) | "
                    f"failures: {failures} | "
                    f"elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
                    flush=True,
                )
                last_progress_time = elapsed

    return failures


def main():
    d = int(sys.argv[1])
    p_index = int(sys.argv[2])
    p = float(p_list[p_index])

    num_workers = os.cpu_count() or 1
    failures = run_simulation(
        d=d,
        p=p,
        num_workers=num_workers,
        total_shots=shots,
        seed_base=seed,
    )

    result = {
        "d": d,
        "p_index": p_index,
        "p": p,
        "shots": shots,
        "failures": failures,
    }

    os.makedirs("result", exist_ok=True)
    output = f"result/result_{d}_{p_index}.json"
    with open(output, "w") as file:
        json.dump(result, file, indent=2)

    print(result)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
