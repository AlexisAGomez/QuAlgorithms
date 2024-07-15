#
# State Preparation Algorithms and Dependencies Rev. 1
#
# Alexis A. Gomez, Dr. Artyom Grigoryan,
# UTSA ECE, March 11, 2023
#
# Reference http://dx.doi.org/10.1117/12.3013940

# Necessary Imports
import numpy as np
from qiskit import QuantumCircuit


def matrix_msob2(y):
    """
    Generates a transformation matrix A from a given vector y, using a series of rotations to align y
    with the first coordinate axis through the Digital signal-induced Heap Transform (DsiHT). This process effectively
    constructs a matrix that can be used to initialize quantum states in certain quantum algorithms.

    Parameters:
    - y: array-like, a vector of values for which the transformation matrix is to be calculated.

    Returns:
    - A: ndarray, the transformation matrix obtained by applying a series of rotations to align
         the input vector y with the first coordinate axis. The matrix A is orthogonal and represents
         the accumulated rotation operations.
    """

    N = len(y)
    A = np.eye(N)
    pi2 = np.pi / 2
    U = np.zeros(N)
    kk = 0
    for i1 in range(N, 1, -1):
        y1 = y[i1 - 2]  # Adjust for 0-based indexing
        yi1 = y[i1 - 1]
        if abs(yi1) > np.finfo(float).eps:
            if y1 == 0:
                u = pi2
            else:
                u = np.arctan2(-yi1, y1)  # Equivalent to acot(-y1/yi1)
            U[i1 - 1] = u
            c1 = np.cos(u)
            s1 = np.sin(u)
            y[i1 - 2] = np.dot([c1, -s1], [y1, yi1])
            S = np.eye(N)
            S[i1 - 2, i1 - 2] = c1
            S[i1 - 1, i1 - 2] = -s1
            S[i1 - 2, i1 - 1] = s1
            S[i1 - 1, i1 - 1] = c1
            A = np.dot(A, S)
            kk += 1
            # print(f'Matrix of rotation # {kk}')
            # print(S)
            # print(f'   rotation by angle {u:.4f} ({u / np.pi * 180:.4f} in D)')
    U[0] = y[0]
    A = A.T
    return A


def get_perm2(N):
    """
    Creates a permutation matrix for arranging elements of a square array into a specific pattern,
    useful in quantum computing operations, particularly for state preparation or algorithmic steps
    requiring specific element ordering.

    Parameters:
    - N: int, the dimension of the square array for which the permutation matrix is generated.
         The matrix size will be N by N^2, facilitating a reshaping or reordering of elements
         in preparation for quantum operations.

    Returns:
    - P: ndarray, the permutation matrix that reorders elements according to the specified pattern,
         useful in quantum computing for state preparation and other algorithmic needs. The matrix
         enables transformation from a linear sequence to a structured form.
    """

    N = float(N)
    init_perm = np.zeros((int(N), int(N * N)))

    j = 0
    for i in range(0, int(N * N), int(np.sqrt(N * N))):
        init_perm[j, i] = 1
        j += 1

    N = int(N) - 1
    P = init_perm

    for i in range(N, 0, -1):
        c = np.roll(init_perm, i, axis=1)
        P = np.vstack([P, c])

    return P


def state_prepare(ImageInput, label='Image '):
    """
    Prepares a quantum state corresponding to a given input image for quantum computing operations.
    The function normalizes the image, applies a unitary transformation based on the matrix_msob2
    function to align the input with quantum computing requirements, and initializes the quantum
    circuit state.

    Parameters:
    - ImageInput: np.ndarray, the input image array to be converted into a quantum state. The image
                  is normalized and used to prepare the quantum state.
    - label: str, optional, default is 'Image '. A label for the quantum operation, providing context
             in the quantum circuit diagram.

    Returns:
    - qc1: QuantumCircuit, the quantum circuit initialized with the quantum state prepared from the
           input image, ready for quantum computing operations.
    """

    qubits = len(ImageInput[0])
    qubits = int(np.log2(qubits))

    # Normalize Image by Energy
    q = ImageInput / np.linalg.norm(ImageInput)

    # Import Matrix for Quantum Circuit Initialization
    # Unitary Matrix from the DsiHT with Strong Wheel Carriage
    # is used here for State Preparation. Provided by Dr. Grigoryan
    ImageMatrixOp = matrix_msob2(q[0])  # Retrieve Unitary Matrix
    ImageMatrixOp = np.transpose(ImageMatrixOp)

    # Create Quantum Circuit with 6 Qubits
    qc1 = QuantumCircuit(qubits, 0)

    # Use 'initialize' for State Preparation. Use ImageInput
    # for State Preparation with 'initialize' normalizing the input.
    qc1.unitary(ImageMatrixOp, range(qubits), label=label)

    return qc1
