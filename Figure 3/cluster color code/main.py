import stim
import time
import sys
import json
import numpy as np
from stimbposd import BPOSD
import gurobipy as gp
from gurobipy import GRB

flag = int(sys.argv[1])
d = int(sys.argv[2])
p_index = int(sys.argv[3])
job_index = int(sys.argv[4])

p_list = np.linspace(3.5e-3,4.5e-3,10)

p = p_list[p_index]

H_d3 = np.array([
    [np.nan, 0, np.nan, 1, 2, 3],
    [2, 3, 4, 5, np.nan, np.nan],
    [1, np.nan, 3, np.nan, 5, 6],
])

H_d5 = np.array([
    [np.nan, 0, np.nan, 1, 2, 3], # green
    [1, np.nan, 3, np.nan, 5, 6], # blue
    [2, 3, 4, 5, 7, 8], # red
    [np.nan, 4, np.nan, 7, 10, 11], # green
    [5, 6, 8, 9, 12, 13], # green
    [10, 11, 14, 15, np.nan, np.nan], # red
    [7, 8, 11, 12, 15, 16], # blue
    [12, 13, 16, 17, np.nan, np.nan], # red
    [9, np.nan, 13, np.nan, 17, 18] # blue
])

H_d7 = np.array([
    [np.nan, 0, np.nan, 1, 2, 3],          # green

    [1, np.nan, 3, np.nan, 5, 6],          # blue

    [2, 3, 4, 5, 7, 8],                    # red

    [np.nan, 4, np.nan, 7, 10, 11],        # green
    [5, 6, 8, 9, 12, 13],                  # green

    [7, 8, 11, 12, 15, 16],                # blue
    [9, np.nan, 13, np.nan, 17, 18],       # blue

    [10, 11, 14, 15, 19, 20],              # red
    [12, 13, 16, 17, 21, 22],              # red

    [np.nan, 14, np.nan, 19, 24, 25],      # green
    [15, 16, 20, 21, 26, 27],              # green
    [17, 18, 22, 23, 28, 29],              # green

    [19, 20, 25, 26, 31, 32],              # blue
    [21, 22, 27, 28, 33, 34],              # blue
    [23, np.nan, 29, np.nan, 35, 36],      # blue

    [24, 25, 30, 31, np.nan, np.nan],      # red
    [26, 27, 32, 33, np.nan, np.nan],      # red
    [28, 29, 34, 35, np.nan, np.nan],      # red
])


H_d9 = np.array([
    [0, np.nan, 1, np.nan, 2, 3],
    
    [np.nan, 1, np.nan, 2, 4, 5],
    
    [2,3,5,6,8,9],
    
    [4,5,7,8,10,11],
    [6,np.nan,9,np.nan,12,13],
    
    [np.nan,7,np.nan,10,14,15],
    [8,9,11,12,16,17],
    
    [10,11,15,16,20,21],
    [12,13,17,18,22,23],
    
    [14,15,19,20,24,25],
    [16,17,21,22,26,27],
    [18,np.nan,23,np.nan,28,29],
    
    [np.nan,19,np.nan,24,30,31],
    [20,21,25,26,32,33],
    [22,23,27,28,34,35],
    
    [24,25,31,32,38,39],
    [26,27,33,34,40,41],
    [28,29,35,36,42,43],
    
    [30,31,37,38,44,45],
    [32,33,39,40,46,47],
    [34,35,41,42,48,49],
    [36,np.nan,43,np.nan,50,51],
    
    [np.nan,37,np.nan,44,52,53],
    [38,39,45,46,54,55],
    [40,41,47,48,56,57],
    [42,43,49,50,58,59],
    
    [44,45,53,54,np.nan,np.nan],
    [46,47,55,56,np.nan,np.nan],
    [48,49,57,58,np.nan,np.nan],
    [50,51,59,60,np.nan,np.nan],
])




def Create_Circuit(H, nb_rounds,flag):
    circuit = stim.Circuit()

    n = int(np.nanmax(H) + 1)
    n_a = H.shape[0]

    for r in range(nb_rounds):

        # data qubit initialization
        if r==0:
            circuit.append('R', range(n))

        #####################
        ### Z stabilizers ###
        #####################
        
        # ancilla qubit initialization
        circuit.append('RX', range(n, n+n_a))
        circuit.append('R', range(n+n_a, n+2*n_a))

        circuit.append('TICK')

        # Bell pair
        for i in range(n_a):
            circuit.append('CX', [n+i, n+n_a+i])

        circuit.append('TICK')

        # first CNOT
        for i in range(n_a):
            if not np.isnan(H[i][4]):
                circuit.append('CX',[int(H[i][4]),n+i])
            if not np.isnan(H[i][5]):
                circuit.append('CX',[int(H[i][5]),n+n_a+i])

        circuit.append('TICK')

        # second CNOT
        for i in range(n_a):
            if not np.isnan(H[i][2]):
                circuit.append('CX',[int(H[i][2]),n+i])
            if not np.isnan(H[i][3]):
                circuit.append('CX',[int(H[i][3]),n+n_a+i])

        circuit.append('TICK')

        # third CNOT
        for i in range(n_a):
            if not np.isnan(H[i][0]):
                circuit.append('CX',[int(H[i][0]),n+i])
            if not np.isnan(H[i][1]):
                circuit.append('CX',[int(H[i][1]),n+n_a+i])

        circuit.append('TICK')

        # undo Bell pair
        for i in range(n_a):
            circuit.append('CX', [n+i, n+n_a+i])

        circuit.append('TICK')

        circuit.append('MX', range(n, n+n_a))
        if flag:
            for i in range(n_a):
                circuit.append('DETECTOR', stim.target_rec(-n_a+i))

        circuit.append('M', range(n+n_a, n+2*n_a))
        if r==0:
            for i in range(n_a):
                circuit.append('DETECTOR', stim.target_rec(-n_a+i))
        else:
            for i in range(n_a):
                circuit.append('DETECTOR', [stim.target_rec(-n_a+i),stim.target_rec(-5*n_a+i)])

        #####################
        ### X stabilizers ###
        #####################

        # ancilla qubit initialization
        circuit.append('RX', range(n, n+n_a))
        circuit.append('R', range(n+n_a, n+2*n_a))

        circuit.append('TICK')

        # Bell pair
        for i in range(n_a):
            circuit.append('CX', [n+i, n+n_a+i])

        circuit.append('TICK')

        # first CNOT
        for i in range(n_a):
            if not np.isnan(H[i][4]):
                circuit.append('CX',[n+i, int(H[i][4])])
            if not np.isnan(H[i][5]):
                circuit.append('CX',[n+n_a+i, int(H[i][5])])

        circuit.append('TICK')

        # second CNOT
        for i in range(n_a):
            if not np.isnan(H[i][2]):
                circuit.append('CX',[n+i, int(H[i][2])])
            if not np.isnan(H[i][3]):
                circuit.append('CX',[n+n_a+i, int(H[i][3])])

        circuit.append('TICK')

        # third CNOT
        for i in range(n_a):
            if not np.isnan(H[i][0]):
                circuit.append('CX',[n+i, int(H[i][0])])
            if not np.isnan(H[i][1]):
                circuit.append('CX',[n+n_a+i, int(H[i][1])])

        circuit.append('TICK')

        # undo Bell pair
        for i in range(n_a):
            circuit.append('CX', [n+i, n+n_a+i])

        circuit.append('TICK')

        circuit.append('MX', range(n, n+n_a))

        if r>0:
            for i in range(n_a):
                circuit.append('DETECTOR', [stim.target_rec(-n_a+i),stim.target_rec(-5*n_a+i)])
        
        circuit.append('M', range(n+n_a, n+2*n_a))
        if flag:
            for i in range(n_a):
                circuit.append('DETECTOR', stim.target_rec(-n_a+i))

        if r < nb_rounds-1:
            circuit.append('TICK')
        else:
            circuit.append('M', range(n))
            for i in range(n_a):
                det_to_add = []
                for j in range(H.shape[1]):
                    if not np.isnan(H[i,j]):
                        det_to_add.append(stim.target_rec(-n+int(H[i,j])))
                det_to_add.append(stim.target_rec(-n-3*n_a+i))
                circuit.append('DETECTOR', det_to_add)
            if H.shape[0] == 3:
                circuit.append(
                    "OBSERVABLE_INCLUDE", 
                    [stim.target_rec(-n+i) for i in [0,2,4]],
                    0
                )
            elif H.shape[0] == 9:
                circuit.append(
                    "OBSERVABLE_INCLUDE", 
                    [stim.target_rec(-n+i) for i in [0,2,4,10,14]],
                    0
                )
            elif H.shape[0] == 18:
                circuit.append(
                    "OBSERVABLE_INCLUDE", 
                    [stim.target_rec(-n+i) for i in [0,2,4,10,14,24,30]],
                    0
                )
            elif H.shape[0] == 30:
                circuit.append(
                    "OBSERVABLE_INCLUDE", 
                    [stim.target_rec(-n+i) for i in [0,2,4,10,14,24,30,37,52]],
                    0
                )

            else:
                raise ValueError("Invalid parity check matrix H")
            
    return circuit





def AddNoise(circuit,p):

    result = stim.Circuit()

    qubit_processed = []

    for instruction_index,instruction in enumerate(circuit):

        if instruction.name == 'RX':
            result.append(instruction)
            result.append("Z_ERROR",[target.qubit_value for target in instruction.targets_copy()],p)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

        elif instruction.name == 'R':
            result.append(instruction)
            result.append("X_ERROR",[target.qubit_value for target in instruction.targets_copy()],p)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

        elif instruction.name == 'MX':
            result.append("Z_ERROR",[target.qubit_value for target in instruction.targets_copy()],p)
            result.append(instruction)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

        elif instruction.name == 'M':
            result.append("X_ERROR",[target.qubit_value for target in instruction.targets_copy()],p)
            result.append(instruction)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

        elif instruction.name == 'CX' or instruction.name == 'CZ':
            result.append(instruction)
            result.append("DEPOLARIZE2",[target.qubit_value for target in instruction.targets_copy()],p)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]
            
        elif instruction.name == 'TICK':
            for i in range(circuit.num_qubits):
                if i not in qubit_processed:
                    result.append("DEPOLARIZE1",i,p)
            result.append(instruction)
            qubit_processed = []

        else:
            result.append(instruction)
            qubit_processed += [target.qubit_value for target in instruction.targets_copy()]

    circuit = result
    
    return circuit



def gurobi_mld_decoder(H, obs, shots, priors):
    num_stabilizers, n = H.shape

    obs = np.asarray(obs).reshape(-1)
    priors = np.asarray(priors).reshape(-1)

    llr_weights = np.log((1.0 - priors) / priors)
    predicted_observables = np.empty((len(shots), 1), dtype=np.int8)

    with gp.Env() as env:
        with gp.Model("MLD_Decoder", env=env) as model:
            model.setParam("OutputFlag", 0)
            model.setParam("Threads", 1)

            e = model.addVars(n, vtype=GRB.BINARY, name="e")
            k = model.addVars(num_stabilizers, vtype=GRB.INTEGER, name="k")

            model.setObjective(
                gp.quicksum(float(llr_weights[j]) * e[j] for j in range(n)),
                GRB.MINIMIZE
            )

            constraints = []
            for i in range(num_stabilizers):
                c = model.addConstr(
                    gp.quicksum(int(H[i, j]) * e[j] for j in range(n)) - 2 * k[i] == 0,
                    name=f"stab_{i}"
                )
                constraints.append(c)

            model.update()

            for shot_idx, s in enumerate(shots):
                if shot_idx%1_000 == 0:
                    print(shot_idx,' / ', len(shots), flush=True)
                s = np.asarray(s).reshape(-1)

                for i in range(num_stabilizers):
                    constraints[i].RHS = int(s[i])

                model.optimize()

                if model.status != GRB.OPTIMAL:
                    raise RuntimeError(
                        f"Gurobi failed for shot {shot_idx}. "
                        f"Status code: {model.status}"
                    )

                flip = int(sum(int(round(e[j].X)) * int(obs[j]) for j in range(n)) % 2)
                predicted_observables[shot_idx, 0] = flip

    return predicted_observables


start = time.time()

batch_size = 10_000

index_d = int((d-3)/2)

H = [H_d3,H_d5,H_d7,H_d9][index_d]

circuit = Create_Circuit(H,nb_rounds=d, flag = flag)
circuit = AddNoise(circuit,p)

sampler = circuit.compile_detector_sampler()
detection_events, observables = sampler.sample(shots = batch_size,separate_observables = True)
        
dem = circuit.detector_error_model(approximate_disjoint_errors=True, decompose_errors = False)
decoder = BPOSD(dem)
H = np.array(decoder._matrices.check_matrix.toarray(), dtype = int)
obs = np.array(decoder._matrices.observables_matrix.toarray(), dtype = int)
priors = decoder._matrices.priors

predicted_observables = gurobi_mld_decoder(H, obs, detection_events, priors)
     
nb_errors = np.sum(~(predicted_observables == observables))

end = time.time()

print(f"Execution time: {end - start} seconds")

result_dict = {}

result_dict['flag'] = flag
result_dict['d'] = d
result_dict['p'] = p
result_dict['job_index'] = job_index
result_dict['nb_shots'] = batch_size
result_dict['nb_mistakes'] = int(nb_errors)

with open(f"Results/result_{flag}_{d}_{p_index}_{job_index}.json", "w") as file:
    json.dump(result_dict, file)