import numpy as np
from typing import List, Union
from numpy.typing import NDArray
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector
from qiskit.primitives import Estimator
from tomologie import diagonalize_pauli_with_circuit


def exact_evolution(
    initial_state: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    time_values: NDArray[np.float_],
    observables: List[SparsePauliOp],
):
    """
    Simulate the exact evolution of a quantum system in state ‘initial_state‘ under a given
    ‘hamiltonian‘ for different ‘time_values‘. The result is a series of expected values
    for given ‘observables‘.

    Args:
    time_values (NDArray[np.float_]): An array of time values.
    initial_state (QuantumCircuit): The circuit preparing the initial quantum state.
    hamiltonian (SparsePauliOp): The Hamiltonien of the system
    observables (List[SparsePauliOp]): The observable to be measure at each the the
    ‘time_values‘.

    Returns:
    NDArray[np.float_]: The expected values of the observable. Should be of shape
    ‘(len(time_values), len(observables))‘.
    """
    hamiltonian_matrix = hamiltonian.to_matrix()
    state_vector = Statevector(initial_state)

    eig_values, eig_vectors = np.linalg.eigh(hamiltonian_matrix)

    omega = np.einsum("s, i ->si", time_values, eig_values)

    evolution_operators = np.exp(-1j * omega)

    computational_evolution_operators = np.einsum(
        "ik, sk, jk -> sij", eig_vectors, evolution_operators, eig_vectors.conj()
    )

    evolved_states = np.einsum(
        "sij, j -> si", computational_evolution_operators, state_vector
    )

    observables_vector = np.stack(
        [observable.to_matrix() for observable in observables]
    )

    observables_expected_values = np.einsum(
        "si, pij, sj -> sp", evolved_states.conj(), observables_vector, evolved_states
    )

    return observables_expected_values


def trotter_evolution(
    initial_state: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    time_values: NDArray[np.float_],
    observables: List[SparsePauliOp],
    num_trotter_steps: NDArray[np.int_],
):
    """
    Simulate, using Trotterisation, the evolution of a quantum system in state ‘initial_state‘
    under a given ‘hamiltonian‘ for different ‘time_values‘. The result is a series of
    expected values for given ‘observables‘.

    Args:
    time_values (NDArray[np.float_]): An array of time values.
    initial_state (QuantumCircuit): The circuit preparing the initial quantum state.
    hamiltonian (SparsePauliOp): The Hamiltonien of the system
    observables (List[SparsePauliOp]): The observable to be measure at each the the
    ‘time_values‘.
    num_trotter_steps: (NDArray[np.int_]): The number of steps of the Trotterization for
    each ‘time_values‘.

    Returns:
    NDArray[np.float_]: The expected values of the observable. Should be of shape
    ‘(len(time_values), len(observables))‘.
    """
    estimator = Estimator()
    circuits = []

    for time_value, num_trotter_step in zip(time_values, num_trotter_steps):

        circuit = trotter_circuit(hamiltonian, time_value, num_trotter_step)
        evolved_state = initial_state.compose(circuit)

        for _ in range(len(observables)):
            circuits.append(evolved_state)

    expected_values = (
        estimator.run(circuits, observables * len(time_values)).result().values
    )

    reshaped_expected_values = np.reshape(
        expected_values, (len(time_values), len(observables))
    )

    return reshaped_expected_values


def evolution_circuit_single_pauli(pauli: Pauli, angle):
    """
    Constructs a quantum circuit representing the evolution under a single Pauli operator.

    Args:
    pauli (Pauli): The Pauli operator for which to construct the evolution circuit.
    angle (float): The rotation angle for the single-qubit rotation.

    Returns:
    QuantumCircuit: The quantum circuit representing the evolution under the given Pauli operator.
    """
    num_qubits = len(pauli)

    qreg = QuantumRegister(num_qubits, "q")
    evolved_circuit = QuantumCircuit(qreg)
    diagonal_circuit = QuantumCircuit(qreg)
    cnot_circuit = QuantumCircuit(qreg)
    rot_circuit = QuantumCircuit(qreg)

    diag_zbits, diagonal_circuit = diagonalize_pauli_with_circuit(pauli)
    z_indices = np.where(diag_zbits)[0]

    for q in range(len(z_indices) - 1):
        cnot_circuit.cx(z_indices[q], z_indices[q + 1])

    rot_circuit.rz(angle, z_indices[-1])

    evolved_circuit = (
        diagonal_circuit.compose(cnot_circuit)
        .compose(rot_circuit)
        .compose(cnot_circuit.inverse())
        .compose(diagonal_circuit.inverse())
    )
    return evolved_circuit


def trotter_circuit(
    hamiltonian: SparsePauliOp,
    total_duration: Union[float, Parameter],
    num_trotter_steps: int,
) -> QuantumCircuit:
    """
    Construct the ‘QuantumCircuit‘ using the first order Trotter formula.

    Args:
    hamiltonian (SparsePauliOp): The Hamiltonian which governs the evolution.
    total_duration (Union[float, Parameter]): The duration of the complete evolution.
    num_trotter_steps (int): The number of trotter steps.

    Returns:
    QuantumCircuit: The circuit of the Trotter evolution operator
    """
    num_qubits = hamiltonian.num_qubits
    qreg = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(qreg)

    total_duration /= num_trotter_steps
    trotter_step_circuit = QuantumCircuit(num_qubits)

    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):

        angle = 2 * total_duration * coeff.real
        trotter_step_circuit.compose(
            evolution_circuit_single_pauli(pauli, angle), inplace=True
        )
    for _ in range(num_trotter_steps):

        circuit.compose(trotter_step_circuit, inplace=True)

    return circuit
