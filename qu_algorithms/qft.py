#
# General 1-D, 2-D QFT Algorithms and Dependencies Rev. 1
#
# Alexis A. Gomez, Dr. Artyom Grigoryan,
# UTSA ECE, March 11, 2023
#

# Necessary Imports
import numpy as np  # This is imported for the numerical/symbolic/fft functionality
from qiskit import QuantumCircuit, transpile  # Allows for creation of Quantum Circuit
from qiskit.quantum_info.operators import Operator  # Allows for creation of A2 Gate
from qiskit.circuit.library.standard_gates import TdgGate  # Imports TdgGate for 1D QFT
from qiskit_aer import Aer

from qu_algorithms.state_preparation import get_perm2, state_prepare  # State Preparation for 2D QFT


def bit_reversal_permutation(n: int, get_matrix=False):
    """
    Generate a bit-reversal permutation for FFT of size n, and optionally print the permutation matrix.

    Parameters:
    - n: int, the size of the FFT, assumed to be a power of 2.
    - print_matrix: bool, if True, prints the permutation matrix.

    Returns:
    - perm: list, the bit-reversed permutation of indices.
    """

    # Ensure n is a power of 2
    if not (n and ((n & (n - 1)) == 0)):
        raise ValueError("n must be a power of 2")

    # Determine the number of bits needed to represent n
    log2_n = n.bit_length() - 1

    # Generate the bit-reversal permutation
    perm = []
    for i in range(n):
        # Reverse the bits of i
        reversed_i = int('{:0{width}b}'.format(i, width=log2_n)[::-1], 2)
        perm.append(reversed_i)

    if get_matrix:
        # Create and print the permutation matrix
        matrix = np.zeros((n, n), dtype=int)
        for i, j in enumerate(perm):
            matrix[i, j] = 1
        return perm, matrix

    return perm


def qft(input_var):
    """
    Performs the Quantum Fourier Transform (QFT) on a given input variable. The function supports both
    quantum circuit initialization from an integer specifying the number of qubits or from a numpy
    array representing the initial state.

    Parameters:
    - input_var: np.ndarray or int. If np.ndarray, it is assumed to be a quantum state for QFT.
                 If int, it specifies the number of qubits for a quantum circuit.

    Returns:
    - QuantumCircuit object after applying QFT and necessary transformations.
    """

    if isinstance(input_var, np.ndarray):
        N = len(input_var[0])
        N = int(np.sqrt(N))
        qc = state_prepare(input_var, label="Input").reverse_bits()
    elif isinstance(input_var, int):
        qc = QuantumCircuit(input_var, 0)
        N = input_var
    else:
        raise Exception('Invalid Input: Enter a np.ndarray or N-number of qubits as an int')

    # Initialize Quantum Circuit
    # qc = QuantumCircuit(N, 0)

    # Gate Definitions
    A2 = Operator([[1, -1], [1, 1]] / np.sqrt(2))
    control_T = TdgGate().control(ctrl_state='0')

    # Recursive 1-D QFT
    qc.unitary(A2, [0], label='A2')

    def recursion(n: int, quantum_circuit):
        if n < N + 1:
            quantum_circuit.csdg(n - 2, n - 1, ctrl_state=0)
            a = 0
            m = n
            while m < N:
                quantum_circuit.append(control_T.power(2 ** (-a)), [m - 2 - a, m])
                m = m + 1
                a = a + 1
            quantum_circuit.unitary(A2, [n - 1], label='A2')
            recursion(n + 1, quantum_circuit=quantum_circuit)

        return quantum_circuit

    # Apply Recursive Algorithm
    qc = recursion(2, qc)

    # Do Permutation
    qc.x(range(N))

    _, Perm = bit_reversal_permutation(int(2 ** N), get_matrix=True)
    Perm = np.asarray(Perm)  # Convert to Numpy Matrix

    # Apply Perm
    qc.unitary(Perm, range(N), label='Perm ')

    return qc.reverse_bits()


def qft2(input_var):
    """
    Implements a two-dimensional Quantum Fourier Transform (2D-QFT) scheme on a given input variable.
    The function initializes a quantum circuit based on either a numpy array representing a quantum
    state or an integer specifying the qubits count for a quantum circuit. It performs 2D-QFT using
    a permutation strategy for handling 2D data.

    Parameters:
    - input_var: np.ndarray or int. If np.ndarray, it is assumed to be a 2D quantum state. If int,
                 it specifies the number of qubits for a quantum circuit, expecting a 2^N x 2^N shape.

    Returns:
    - QuantumCircuit object after applying 2D-QFT and necessary permutations.
    """

    if isinstance(input_var, np.ndarray):
        N = len(input_var[0])
        N = int(np.sqrt(N))
        qubits = len(input_var[0])
        qubits = int(np.log2(qubits))
        qc1 = state_prepare(input_var)
    elif isinstance(input_var, int):
        qubits = int(input_var * 2)
        N = int(np.power(2, qubits / 2))
        qc1 = QuantumCircuit(qubits, 0)
    else:
        raise Exception('Invalid Input: Enter a np.ndarray or NxN-number of qubits as an int')

    # Implement 2-D QFT Quantum Scheme
    # This section implements the 2-D QFT using 2 N/2-qubit QFTs. The
    # first stage applies the QFT to the first dimension. Afterward,
    # a permutation is done in order to prepare for the second stage.
    # The second stage applies the QFT to the second dimension. Finally,
    # the permutation is done again to align the values as the initial input.

    # Apply the N/2-qubit QFT (Column)
    qft1 = qft(int(qubits / 2)).to_gate(label=" QFT ")
    qc1.append(qft1, range(int(qubits / 2)))

    # Creating Permutation
    Perm = get_perm2(N)

    # Convert to Numpy Matrix
    Perm = np.asarray(Perm)

    # Apply Perm
    qc1.unitary(Perm, range(qubits), label='Perm ')

    # Apply the N/2-qubit QFT (Row)
    qc1.append(qft1, range(int(qubits / 2)))

    # Apply Perm
    qc1.unitary(np.transpose(Perm), range(qubits), label='Perm ')

    return qc1


def _get_theoretical_sv(circuit, reverse_bits=False):
    """
    Calculates the theoretical state vector for a given quantum circuit. It supports the option to
    reverse the bits of the quantum circuit before computing the state vector.

    Parameters:
    - circuit: QuantumCircuit, the quantum circuit to compute the state vector for.
    - reverse_bits: bool, optional, default is False. If True, reverses the bits of the circuit before
                    computing the state vector.

    Returns:
    - tuple: (qc_state, qc_state_dist) where qc_state is the state vector and qc_state_dist is the
             distribution (list) of the state vector's values.
    """

    if reverse_bits:
        circuit = circuit.reverse_bits()

    svsim = Aer.get_backend('statevector_simulator')

    job = svsim.run(transpile(circuit.remove_final_measurements(inplace=False), svsim))
    qc_state = job.result().get_statevector(circuit)

    qc_state_dist = qc_state.to_dict()
    qc_state_dist = list(qc_state_dist.values())
    qc_state_dist = np.array(qc_state_dist)

    return qc_state, qc_state_dist


def verify_qft(input_var: np.ndarray, qc: QuantumCircuit, print_bool=False):
    """
    Verifies the Quantum Fourier Transform (QFT) results against classical 1D-DFT for a given input
    numpy array and a quantum circuit. It prints the verification result if specified.

    Parameters:
    - input_var: np.ndarray, the input array representing the quantum state to be verified.
    - qc: QuantumCircuit, the quantum circuit that has performed QFT on the input_var.
    - print_bool: bool, optional, default is False. If True, prints whether the 1D-DFT and 1D-QFT
                 results match.

    Returns:
    - bool: True if the classical 1D-DFT and the quantum 1D-QFT results match, False otherwise.
    """

    # Get Theoretical State Vector
    _, qc_state_dist = _get_theoretical_sv(qc)
    qc_state_dist = np.reshape(qc_state_dist, (1, len(input_var[0])))

    # Verification of Results
    # This section computes the 2-D DFT of the
    # input for the previous Quantum Circuit.

    # Convert q to NxN Image
    q = input_var / np.linalg.norm(input_var)

    # Compute 1-D DFT
    c_fft = np.fft.fft(q) / np.sqrt(len(input_var[0]))

    # Verify Classical 2D-DFT with 2D-QFT
    verify = np.allclose(c_fft, qc_state_dist)
    if print_bool:
        print(f"1D-DFT == 1D-QFT: {verify}")

    return verify


def verify_qft2(input_var: np.ndarray, qc: QuantumCircuit, print_bool=False):
    """
    Verifies the two-dimensional Quantum Fourier Transform (2D-QFT) results against classical 2D-DFT
    for a given input numpy array and a quantum circuit. It prints the verification result if
    specified.

    Parameters:
    - input_var: np.ndarray, the input array representing the 2D quantum state to be verified.
    - qc: QuantumCircuit, the quantum circuit that has performed 2D-QFT on the input_var.
    - print_bool: bool, optional, default is False. If True, prints whether the 2D-DFT and 2D-QFT
                 results match.

    Returns:
    - bool: True if the classical 2D-DFT and the quantum 2D-QFT results match, False otherwise.
    """

    N = len(input_var[0])
    N = int(np.sqrt(N))

    # Get Theoretical State Vector
    _, qc_state_dist = _get_theoretical_sv(qc)

    # Verification of Results
    # This section computes the 2-D DFT of the
    # input for the previous Quantum Circuit.

    # Convert q to NxN Image
    q = input_var / np.linalg.norm(input_var)
    q = np.reshape(q, (N, -1))

    # Compute NxN 2-D DFT
    c_fft = np.fft.fft2(q) / np.sqrt(len(input_var[0]))

    # Convert c_fft to 1D Matrix
    c_fft = c_fft.flatten()
    # c_fft = np.conj(c_fft)

    # Verify Classical 2D-DFT with 2D-QFT
    verify = np.allclose(c_fft, qc_state_dist)
    if print_bool:
        print(f"2D-DFT == 2D-QFT: {verify}")

    return verify
