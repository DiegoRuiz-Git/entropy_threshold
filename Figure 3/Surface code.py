import stim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def H2(p):
    if p <= 0 or p>=1:
        return 0
    return - p * np.log2(p) - (1-p) * np.log2(1-p)


def H_E_pheno(p, q):
    H_tot = 4*1/2*H2(p) + 2*1/2*H2(q)
    return H_tot


def H_stab_pheno(p, q):
    prob_product = ((1 - 2*p)**4) * ((1 - 2*q)**2)
    return H2(0.5 * (1 - prob_product))


def Create_Ancilla_type(dz,dx):
    Ancilla_type = np.full((dz+1,dx+1),' ')

    for i in range(dz+1):
        for j in range(dx+1):
            if (j+i)%2 == 1 and j>0 and j<dx:
                Ancilla_type[i,j] = 'Z'
            if (j+i)%2 == 0 and i>0 and i<dz:
                Ancilla_type[i,j] = 'X'

    return Ancilla_type


def Create_Ancilla_Data_number(dz,dx):
    
    Ancilla_type = Create_Ancilla_type(dz,dx)
    
    Ancilla_number = np.full((dz+1,dx+1),-1,dtype = int)
    Data_number = np.zeros((dz,dx),dtype = int)

    compt = 0
    for i in range(dz+1):
        # ancilla row
        for j in range(dx+1):
            if Ancilla_type[i,j] != ' ':
                Ancilla_number[i,j] = compt
                compt += 1
        # data row
        if i < dz:
            for j in range(dx):
                Data_number[i,j] = compt
                compt += 1
                
    return Ancilla_number, Data_number


def Create_Ancilla_Data_counter(dz,dx):
    
    Ancilla_type = Create_Ancilla_type(dz,dx)
    
    Ancilla_counter = np.full((dz+1,dx+1),-1,dtype = int)
    Data_counter = np.zeros((dz,dx),dtype = int)

    compt = 0
    for i in range(Ancilla_type.shape[0]):
        # ancilla row
        for j in range(Ancilla_type.shape[1]):
            if Ancilla_type[i,j] != ' ':
                Ancilla_counter[i,j] = compt
                compt += 1

    compt = 0
    for i in range(Ancilla_type.shape[0]):        
        # data row
        if i < Ancilla_type.shape[0]-1:
            for j in range(dx):
                Data_counter[i,j] = compt
                compt += 1
                
    return Ancilla_counter,Data_counter



def CreateCircuit_Z(dz,dx,rounds):

    Ancilla_type = Create_Ancilla_type(dz,dx)
    Ancilla_number,Data_number = Create_Ancilla_Data_number(dz,dx)
    Ancilla_counter, Data_counter = Create_Ancilla_Data_counter(dz,dx)

    circuit = stim.Circuit()

    rounds = rounds

    for r in range(rounds):
        
        # data prep
        if r==0:
            circuit.append('R',Data_number.flatten().tolist())

        # ancilla prep
        for i in range(dz+1):
            for j in range(dx+1):
                if Ancilla_type[i,j] == 'Z':
                    circuit.append('R',Ancilla_number[i,j])
                if Ancilla_type[i,j] == 'X':
                    circuit.append('RX',Ancilla_number[i,j])

        circuit.append('TICK')

        # first CNOT (upper left)
        for i in range(1,dz+1):
            for j in range(1,dx+1):
                if Ancilla_type[i,j] == 'Z':
                    circuit.append('CX',[Data_number[i-1,j-1],Ancilla_number[i,j]])
                if Ancilla_type[i,j] == 'X':
                    circuit.append('CX',[Ancilla_number[i,j],Data_number[i-1,j-1]])

        circuit.append('TICK')

        # second CNOT (Z upper right, X lower left)
        for i in range(1,dz+1):
            for j in range(1,dx+1):
                if Ancilla_type[i,j] == 'Z':
                    circuit.append('CX',[Data_number[i-1,j],Ancilla_number[i,j]])
                if Ancilla_type[i,j] == 'X':
                    circuit.append('CX',[Ancilla_number[i,j],Data_number[i,j-1]])
                  
        circuit.append('TICK')

        # third CNOT (Z lower left, X upper right)
        for i in range(dz):
            for j in range(dx):
                if Ancilla_type[i,j] == 'Z':
                    circuit.append('CX',[Data_number[i,j-1],Ancilla_number[i,j]])
                if Ancilla_type[i,j] == 'X':
                    circuit.append('CX',[Ancilla_number[i,j],Data_number[i-1,j]])

        circuit.append('TICK')

        # fourth CNOT (lower right)
        for i in range(dz):
            for j in range(dx):
                if Ancilla_type[i,j] == 'Z':
                    circuit.append('CX',[Data_number[i,j],Ancilla_number[i,j]])
                if Ancilla_type[i,j] == 'X':
                    circuit.append('CX',[Ancilla_number[i,j],Data_number[i,j]])

        circuit.append('TICK')

        # ancilla measurement
        for i in range(dz+1):
            for j in range(dx+1):
                if Ancilla_type[i,j] == 'Z':
                    circuit.append('MZ',Ancilla_number[i,j])
                if Ancilla_type[i,j] == 'X':
                    circuit.append('MX',Ancilla_number[i,j])

        # detector first round
        if r==0:
            for i in range(dz+1):
                for j in range(dx+1):
                    if Ancilla_type[i,j] == 'Z':
                        circuit.append("DETECTOR",[stim.target_rec(-dx*dz+1+Ancilla_counter[i,j])])

        # detector next rounds
        if r>=1:
            for i in range(dx*dz-1):
                circuit.append("DETECTOR",[stim.target_rec(-2*dx*dz+2+i),stim.target_rec(-dx*dz+1+i)])
                                   
        # data measurement
        if r==rounds-1:
            circuit.append('M',Data_number.flatten().tolist())
            # final detector
            for i in range(dz+1):
                for j in range(dx+1):
                    if Ancilla_type[i,j] == 'Z':
                        data_qubits_stab = [Data_counter[x,y] for x in [i-1,i] for y in [j-1,j] 
                                            if (x>=0 and y>=0 and x<dz and y<dx)]
                        circuit.append("DETECTOR",[stim.target_rec(-dx*dz+x) for x in data_qubits_stab]
                                       + [stim.target_rec(-2*dx*dz + 1 + Ancilla_counter[i,j])])

            # observable
            circuit.append("OBSERVABLE_INCLUDE",[stim.target_rec(-dx*dz+i) for i in Data_counter[:,0]],0)

        else:
            circuit.append('TICK')
            
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

        elif instruction.name == 'CX':
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

# H(E) for circuit-level

d = 5
p = 1e-3
circuit = CreateCircuit_Z(d, d, d)
circuit = AddNoise(circuit, p)
dem = circuit.detector_error_model()

errors = []
nb_stab_shared = []

for instruction in dem:
    num_detectors = 0
    error_part_stab = False
    for detector in instruction.targets_copy():
        if str(detector)[0] == 'D':
            # detector in the bulk of the circuit
            if detector.val == 45:
                error_part_stab = True
                num_detectors+=1
            else:
                num_detectors+=1
    if error_part_stab:
        errors.append(instruction.args_copy()[0])
        nb_stab_shared.append(num_detectors)
        
        
def H_E_circuit(p):
    H_tot = 0
    for index,error in enumerate(errors):
        H_tot += 1/nb_stab_shared[index]*H2(error*p/1e-3)

    return H_tot

# H(S) for circuit-level

syndrome_errors = []

for instruction in dem:
    for detector in instruction.targets_copy():
        if str(detector)[0] == 'D':
            if detector.val == 45:
                syndrome_errors.append(instruction.args_copy()[0])

syndrome_errors = np.array(syndrome_errors)

def H_stab_circuit(p):
    return H2(0.5 * (1 - np.prod(1 - 2 * syndrome_errors*p/1e-3)))  


matplotlib.use('Agg')
plt.style.use('rgplot')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure(figsize = (3,2))

plt.plot([], [], '-', label = 'circuit-level', c='k')
plt.plot([], [], '--', label = 'pheno.', c='k')

p_list = np.logspace(-2.2,-1,100)


plt.plot(p_list, [H_stab_pheno(p,p) for p in p_list], '-', c=colors[0])
plt.plot(p_list, [H_E_pheno(p,p) for p in p_list],'-', c=colors[0])

plt.plot(p_list,[H_stab_circuit(p) for p in p_list],  label = r'$H(S)$', c=colors[1])
plt.plot(p_list,[H_E_circuit(p) for p in p_list], label = r'$H(E)$', c=colors[1])
plt.plot([0.033,0.033], [0.4,1], '-', c='k')
plt.fill_between([0.015, 0.018], 0.6, 1.5, color='k', alpha=0.2)
plt.plot([0.015, 0.015], [0.6, 1.5], '-', c='k')
plt.plot([0.018, 0.018], [0.6, 1.5], '-', c='k')




plt.loglog()
plt.ylabel('entropy', fontsize = 7)
plt.xlabel(r'$p$', fontsize = 9)
textes_legende = plt.legend().get_texts()
textes_legende[0].set_fontsize(7)
textes_legende[1].set_fontsize(7)
textes_legende[2].set_fontsize(9)
textes_legende[3].set_fontsize(9)

plt.xticks(fontsize = 7)
plt.yticks(fontsize = 7)
plt.xlim((10**(-2.2),10**(-1)))

plt.savefig('surface code.pdf')



