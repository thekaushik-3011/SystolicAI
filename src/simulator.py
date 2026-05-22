"""
Cycle-accurate simulator engine.
"""
from src.array import SystolicArray
from src.workloads import skew_matrices
import numpy as np

class Simulator:
    def __init__(self, array_rows, array_cols):
        self.array_rows = array_rows
        self.array_cols = array_cols
        self.array = SystolicArray(array_rows, array_cols)
        
        self.total_cycles = 0
        self.history = []  # To store activity per cycle
        
    def run(self, A, B):
        """
        Run the simulation for matrix multiplication C = A x B.
        Handles tiling if A and B are larger than the array.
        """
        M, K = A.shape
        K_b, N = B.shape
        assert K == K_b, "Inner dimensions must match"
        
        # Initialize output matrix
        C = np.zeros((M, N))
        
        # Tile execution
        for m in range(0, M, self.array_rows):
            for n in range(0, N, self.array_cols):
                # Get the slices
                A_slice = A[m:m+self.array_rows, :]
                B_slice = B[:, n:n+self.array_cols]
                
                # Pad to array dimensions if necessary
                A_padded = np.zeros((self.array_rows, K))
                B_padded = np.zeros((K, self.array_cols))
                
                actual_rows = A_slice.shape[0]
                actual_cols = B_slice.shape[1]
                
                A_padded[:actual_rows, :] = A_slice
                B_padded[:, :actual_cols] = B_slice
                
                # Skew matrices for systolic inputs
                skewed_A, skewed_B = skew_matrices(A_padded, B_padded)
                
                num_cycles = skewed_A.shape[0]
                
                self.array.reset()
                
                for cycle in range(num_cycles):
                    act_in = skewed_A[cycle, :].tolist()
                    wt_in = skewed_B[cycle, :].tolist()
                    
                    self.array.step(act_in, wt_in)
                    self.total_cycles += 1
                    
                    # Record activity for visualization/metrics
                    self.history.append(self.array.get_pe_activity())
                    
                # Extract results and place in C
                partial_sums = self.array.get_partial_sums()
                for r in range(actual_rows):
                    for c in range(actual_cols):
                        C[m+r, n+c] = partial_sums[r][c]
                        
        return C
