"""
Workload generation and matrix skewing.
"""
import numpy as np

def generate_matrices(M, K, N, seed=None):
    """
    Generate synthetic random matrices for testing.
    """
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randint(1, 10, size=(M, K)).astype(float)
    B = np.random.randint(1, 10, size=(K, N)).astype(float)
    return A, B

def skew_matrices(A, B):
    """
    Skew the input matrices so they flow correctly into the systolic array.
    
    A flows left-to-right. Element A[r, k] arrives at PE(r, 0) at cycle r + k.
    B flows top-to-bottom. Element B[k, c] arrives at PE(0, c) at cycle c + k.
    
    Returns:
        skewed_A: numpy array of shape (total_cycles, M)
        skewed_B: numpy array of shape (total_cycles, N)
    """
    M, K = A.shape
    K_b, N = B.shape
    if K != K_b:
        raise ValueError(f"Inner dimensions must match: {K} != {K_b}")
        
    total_cycles = M + N + K - 2
    
    # +1 because cycles are 0-indexed and max cycle is `total_cycles`
    num_cycles = total_cycles + 1
    
    skewed_A = np.zeros((num_cycles, M))
    skewed_B = np.zeros((num_cycles, N))
    
    for r in range(M):
        for k in range(K):
            cycle = r + k
            skewed_A[cycle, r] = A[r, k]
            
    for c in range(N):
        for k in range(K):
            cycle = c + k
            skewed_B[cycle, c] = B[k, c]
            
    return skewed_A, skewed_B
