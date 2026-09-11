import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import stim
import json
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve

matplotlib.use('Agg')

# determine the crossing threshold
def determine_crossing(logical_error,p_list):
    spline0 = UnivariateSpline(p_list, logical_error[0], s=0.5)
    spline1 = UnivariateSpline(p_list, logical_error[1], s=0.5)
    
    def f(p):
        if p<=max(p_list) and p>=min(p_list):
            return spline0(p) - spline1(p)
        else:
            return 1
    
    pth = fsolve(f, x0 = p_list[int(len(p_list)/2)])[0]

    return pth

# d=7 Color code parity check matrix
H_d7 = np.array([
    [np.nan, 0, np.nan, 1, 2, 3],
    [1, np.nan, 3, np.nan, 5, 6],
    [2, 3, 4, 5, 7, 8],
    [np.nan, 4, np.nan, 7, 10, 11],
    [5, 6, 8, 9, 12, 13],
    [7, 8, 11, 12, 15, 16],
    [9, np.nan, 13, np.nan, 17, 18],
    [10, 11, 14, 15, 19, 20],
    [12, 13, 16, 17, 21, 22],
    [np.nan, 14, np.nan, 19, 24, 25],
    [15, 16, 20, 21, 26, 27],
    [17, 18, 22, 23, 28, 29],
    [19, 20, 25, 26, 31, 32],
    [21, 22, 27, 28, 33, 34],
    [23, np.nan, 29, np.nan, 35, 36],
    [24, 25, 30, 31, np.nan, np.nan],
    [26, 27, 32, 33, np.nan, np.nan],
    [28, 29, 34, 35, np.nan, np.nan],
])

# Color code circuit
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
            if H.shape[0] == 18:
                circuit.append(
                    "OBSERVABLE_INCLUDE", 
                    [stim.target_rec(-n+i) for i in [0,2,4,10,14,24,30]],
                    0
                )
            else:
                raise ValueError("Invalid parity check matrix H")
            
    return circuit

# depolarizing noise
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


# Load gurobi decoding
# no flag circuit

d_list = np.arange(3,11,2)
p_list = np.linspace(3.5e-3,4.5e-3,10)

Logical_Error = np.zeros((len(d_list), len(p_list)))

for index_d, d in enumerate(d_list):
    for index_p, p in enumerate(p_list):
        nb_errors = 0
        nb_shots = 0

        for j in range(10):
            with open(f"./cluster color code/Results/result_{0}_{d}_{index_p}_{j}.json", "r") as f:
                data = json.load(f)

            nb_errors += data["nb_mistakes"]
            nb_shots += data["nb_shots"]

        Logical_Error[index_d, index_p] = nb_errors/nb_shots
        
p_th_noflag = determine_crossing(Logical_Error[-2:],p_list)

# Load gurobi decoding
# flag circuit

p_list = np.linspace(4e-3,7e-3,10)

Logical_Error = np.zeros((len(d_list), len(p_list)))

for index_d, d in enumerate(d_list):
    for index_p, p in enumerate(p_list):
        nb_errors = 0
        nb_shots = 0

        for j in range(10):
            try:
                with open(f"./cluster color code/Results/result_{1}_{d}_{index_p}_{j}.json", "r") as f:
                    data = json.load(f)

                nb_errors += data["nb_mistakes"]
                nb_shots += data["nb_shots"]
            except:
                try:
                    for k in range(10):
                        with open(f"./cluster color code/Results/result_{1}_{d}_{index_p}_{j}_{k}.json", "r") as f:
                            data = json.load(f)

                        nb_errors += data["nb_mistakes"]
                        nb_shots += data["nb_shots"]
                except:
                    pass
                                
        if nb_shots!=0:
            Logical_Error[index_d, index_p] =  nb_errors/nb_shots
        else:
            Logical_Error[index_d, index_p] = np.nan
            
p_th_flag = determine_crossing(Logical_Error[-2:,:7],p_list[:7])

## entropy no flag

circuit = Create_Circuit(H_d7, nb_rounds = 7, flag = False)
circuit = AddNoise(circuit, p = 1e-3)
dem = circuit.detector_error_model(approximate_disjoint_errors = True, decompose_errors = False)

# error entropy

errors = []
nb_stab_shared = []

for instruction in dem:
    num_detectors = 0
    error_part_stab = False
    for detector in instruction.targets_copy():
        if detector.is_relative_detector_id:
            # detector in the bulk of the circuit
            if detector.val == 23:
                error_part_stab = True
                num_detectors+=1
            else:
                num_detectors+=1
    if error_part_stab:
        errors.append(instruction.args_copy()[0])
        nb_stab_shared.append(num_detectors)
        
def H_E(p, errors, nb_stab_shared):
    H_tot = 0
    for index,error in enumerate(errors):
        H_tot += 1/nb_stab_shared[index]*H2(error*p/1e-3)

    return H_tot

# syndrome entropy

syndrome_errors = []

for instruction in dem:
    for detector in instruction.targets_copy():
        if detector.is_relative_detector_id:
            if detector.val == 23:
                syndrome_errors.append(instruction.args_copy()[0])

syndrome_errors = np.array(syndrome_errors)

def H2(p):
    if p <= 0 or p>=1:
        return 0
    return - p * np.log2(p) - (1-p) * np.log2(1-p)

def H_stab(p, syndrome_errors):
    return H2(0.5 * (1 - np.prod(1 - 2 * syndrome_errors*p/1e-3)))

plt.style.use('rgplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize = (3,2))
plt.plot([],[],'-', c='k', label = 'with flag qubits')
plt.plot([],[],'-', c='k', label = 'w/o flag qubits')
p_list = np.logspace(np.log10(3e-3),np.log10(1e-2),50)
plt.plot(p_list,[H_stab(p, syndrome_errors) for p in p_list], label = r'$H(S)$', c=colors[0])
plt.plot(p_list,[H_E(p, errors, nb_stab_shared) for p in p_list], label = r'$H(E)$', c=colors[0])
plt.plot([p_th_noflag,p_th_noflag],[6e-1,8e-1],'-', c='k')
plt.ylabel('entropy', fontsize = 7)
plt.xlabel(r'$p$', fontsize = 9)
plt.loglog()

## entropy flag

circuit = Create_Circuit(H_d7, nb_rounds = 7, flag = True)
circuit = AddNoise(circuit, p = 1e-3)
dem = circuit.detector_error_model(approximate_disjoint_errors = True, decompose_errors = False)

# error entropy

errors_stab = []
nb_stab_shared_stab = []

for instruction in dem:
    num_detectors = 0
    error_part_stab = False
    for detector in instruction.targets_copy():
        if detector.is_relative_detector_id:
            # detector in the bulk of the circuit
            if detector.val == 77:
                error_part_stab = True
                num_detectors+=1
            else:
                num_detectors+=1
    if error_part_stab:
        errors_stab.append(instruction.args_copy()[0])
        nb_stab_shared_stab.append(num_detectors)
        
errors_flag = []
nb_stab_shared_flag = []

for instruction in dem:
    num_detectors = 0
    error_part_stab = False
    for detector in instruction.targets_copy():
        if detector.is_relative_detector_id:
            # flag in the bulk of the circuit
            if detector.val == 59:
                error_part_stab = True
                num_detectors+=1
            else:
                num_detectors+=1
    if error_part_stab:
        errors_flag.append(instruction.args_copy()[0])
        nb_stab_shared_flag.append(num_detectors)
        
# syndrome entropy

syndrome_errors = [[],[],[]]

for instruction in dem:
    is_det_77 = False
    is_det_59 = False
    for detector in instruction.targets_copy():
        if detector.is_relative_detector_id:
            if detector.val == 77:
                is_det_77 = True
            elif detector.val == 59:
                is_det_59 = True
    if is_det_77 and is_det_59:
        syndrome_errors[2].append(instruction.args_copy()[0])
    elif is_det_77:
        syndrome_errors[0].append(instruction.args_copy()[0])
    elif is_det_59:
        syndrome_errors[1].append(instruction.args_copy()[0])
        

def H_stab(p, syndrome_errors):
    proba_10 = 0.5 * (1 - np.prod(1 - 2 * np.array(syndrome_errors[0])*p/1e-3))
    proba_01 = 0.5 * (1 - np.prod(1 - 2 * np.array(syndrome_errors[1])*p/1e-3))
    proba_11 = 0.5 * (1 - np.prod(1 - 2 * np.array(syndrome_errors[2])*p/1e-3))
    proba_00 = 1 - proba_10 - proba_01 - proba_11
    return (
        - proba_00 * np.log2(proba_00)
        - proba_01 * np.log2(proba_01)
        - proba_11 * np.log2(proba_11)
        - proba_10 * np.log2(proba_10)
    )

plt.plot(p_list,[H_stab(p, syndrome_errors) for p in p_list], c = colors[1])
plt.plot(p_list,[H_E(p, errors_stab, nb_stab_shared_stab) + H_E(p, errors_flag, nb_stab_shared_flag) for p in p_list], c = colors[1])
plt.plot([p_th_flag,p_th_flag],[0.8,1.2],'--', c='k')
plt.legend(loc = 'upper left')
textes_legende = plt.legend().get_texts()
textes_legende[0].set_fontsize(7)
textes_legende[1].set_fontsize(7)
textes_legende[2].set_fontsize(9)
textes_legende[3].set_fontsize(9)

for label in plt.gca().get_yticklabels(which='both'):
    label.set_fontsize(7)
for label in plt.gca().get_xticklabels(which='both'):
    label.set_fontsize(7)

plt.xlim((3e-3,1e-2))
plt.savefig('color code.pdf')
