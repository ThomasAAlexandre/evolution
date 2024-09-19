from typing import Tuple, List
from numpy.typing import NDArray
import numpy as np
import scipy as sp
from qiskit import *
from qiskit.quantum_info import Pauli, PauliList
from qiskit import quantum_info
from qiskit.primitives import Sampler


def bitstring_to_bits(bit_string: str) -> NDArray[np.bool_]:
    """
    Turns a string of bits into an array of bits

    Args:
        bit_string (str): string of bits

    Returns:
        bits (NDArray): array of bits
    """
    bit_string = reversed(str(bit_string))
    bits = np.array([i == "1" for i in bit_string])
    return bits


def diag_pauli_expectation_value(pauli: Pauli, counts: dict) -> float:
    """
    Estimates average value of the execution of a diagonal pauli chain on a circuit

    Args:
        pauli (Pauli): string of paulis
        counts (dict): dictionary of counts from execution

    Returns:
        expectation_value (float): estimated value of the execution of a diagonal pauli chain on a circuit
    """
    expectation_value = 0.0
    n_counts = 0
    zVec = pauli.z.astype(int)

    for key in counts:
        n_counts += counts[key]

        expectation_value += (
            (-1) ** (np.dot(zVec, bitstring_to_bits(key).astype(int)) % 2)
        ) * counts[key]

    return expectation_value / n_counts


def diag_circuit(pauli: Pauli) -> QuantumCircuit:
    """
    Creates transformation circuit for a pauli string

    Args:
        pauli (Pauli): string of paulis

    Returns:
        circuit (QuantumCircuit): transformation circuit for a pauli string
    """
    qreg = QuantumRegister(len(pauli.x), "q")
    circuit = QuantumCircuit(qreg)
    for i in range(len(pauli.z)):
        if pauli.x[i]:
            if pauli.z[i]:
                circuit.sdg(qreg[i])
            circuit.h(qreg[i])
    return circuit


def diagonalize_pauli_with_circuit(
    pauli: Pauli,
) -> tuple[Pauli, QuantumCircuit]:
    """
    Creates transformation circuit and diagonalizes a pauli string

    Args:
        pauli (Pauli): string of paulis

    Returns:
        diag_pauli (Pauli): string of diagonal paulis
        circuit (QuantumCircuit): transformation circuit for a pauli string
    """
    diag_zbits = np.logical_or(np.array(pauli.z), np.array(pauli.x))
    diag_xbits = np.zeros(len(diag_zbits))
    diagonal_pauli = Pauli((diag_zbits, diag_xbits))
    circuit = diag_circuit(pauli)
    assert np.all(~diagonal_pauli.x)
    return diagonal_pauli, circuit


def estimate_expectation_values(
    paulis: PauliList,
    state_circuit: QuantumCircuit,
    num_shots: int,
) -> NDArray[np.float_]:
    """
    Estimate average values for multiple pauli chains

    Args:
        paulis (PauliList): list of pauli strings
        state_circuit (QuantumCircuit): circuit associated with the state we want to find
        num_shots (int): number of shots for running quantum circuits

    Returns:
        expectation_values (NDArray): Array of estimated average values for each circuit
    """
    diag_paulis = []
    circuits = []
    state_circuit_with_circuits = []

    for i in range(len(paulis)):
        diag_pauli, diag_circuit = diagonalize_pauli_with_circuit(paulis[i])
        diag_paulis.append(diag_pauli)
        circuits.append(diag_circuit)

    for i in range(len(circuits)):
        new_circuit = state_circuit.compose(circuits[i])
        new_circuit.measure_all()
        state_circuit_with_circuits.append(new_circuit)

    job = Sampler().run(state_circuit_with_circuits, shots=num_shots)

    expectation_values = np.zeros(len(paulis))
    for i in range(len(paulis)):
        pauli_to_diag = paulis[i]
        diag_zbits = np.logical_or(np.array(pauli_to_diag.z), np.array(pauli_to_diag.x))
        diag_xbits = np.zeros(len(diag_zbits))
        diag_pauli = Pauli((diag_zbits, diag_xbits))
        expectation_values[i] = diag_pauli_expectation_value(
            diag_pauli, job.result().quasi_dists[i].binary_probabilities()
        )

    return expectation_values


def state_tomography(
    state_circuit: QuantumCircuit, num_shots: int
) -> NDArray[np.complex_]:
    """
    Find unknown state vector associated to a circuit

    Args:
        state_circuit (QuantumCircuit): circuit associated with the state we want to find
        num_shots (int): number of shots for running quantum circuits

    Returns:
        state_vector (NDArray): Array with amplitudes associated with unknown state vector
    """
    number_qubits = state_circuit.num_qubits
    all_paulis = quantum_info.pauli_basis(number_qubits)

    eev = (estimate_expectation_values(all_paulis, state_circuit, num_shots)) / (
        2**number_qubits
    )

    pauli_matrices = all_paulis.to_matrix()
    density_matrix = np.zeros((2**number_qubits, 2**number_qubits), dtype=np.complex_)
    for i in range(0, len(pauli_matrices)):
        density_matrix += eev[i] * pauli_matrices[i]

    eig_vals, state_vector = sp.linalg.eigh(density_matrix)
    state_vector = state_vector[:, np.argmax(eig_vals)]

    return state_vector
