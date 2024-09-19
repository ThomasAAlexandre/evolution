from Trotterisation import trotter_evolution, exact_evolution
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.quantum_info import PauliList, SparsePauliOp, Pauli
import numpy as np
from qiskit import *


def main():
    # 1 spin hamiltonian
    pauli_list = PauliList(["X", "Y"])
    coeff = np.array([(-(0.6) * np.cos(np.pi / 4)), (-(0.6) * np.sin(np.pi / 4))])
    hamiltonian1 = SparsePauliOp(pauli_list, coeff)

    # 2 spins hamiltonian
    term1 = (0.20) * (
        SparsePauliOp(Pauli("XX"))
        + SparsePauliOp(Pauli("YY"))
        + SparsePauliOp(Pauli("ZZ"))
    )
    term2 = -(1.05) * SparsePauliOp(Pauli("IZ"))
    term3 = -(0.95) * SparsePauliOp(Pauli("ZI"))
    hamiltonian2 = term1 + term2 + term3

    time_values = np.arange(0, 60, 0.2)
    observables = [SparsePauliOp("Z"), SparsePauliOp("X"), SparsePauliOp("Y")]
    num_trotter_steps = np.arange(0, 300, 1)
    num_qubits = hamiltonian1.num_qubits

    qreg = QuantumRegister(num_qubits)
    initial_state = QuantumCircuit(qreg)

    initial_state.h(qreg)

    exact_values = exact_evolution(
        initial_state, hamiltonian1, time_values, observables
    )

    trotterized_values = trotter_evolution(
        initial_state, hamiltonian1, time_values, observables, num_trotter_steps
    )

    for i, observable in enumerate(exact_values.T):
        plt.plot(time_values, observable, label=f"Observable {i+1} (Exact)")

    for i, observable in enumerate(trotterized_values.T):
        plt.plot(time_values, observable, label=f"Observable {i+1} (Trotterized)")

    plt.xlabel("Time")
    plt.ylabel("Expected Values")
    plt.title("Comparison between exact and Trotterized evolution")
    plt.legend()
    plt.show()

    return "SUCCESS"


result = main()
print(result)
