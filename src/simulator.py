"""
Cycle-accurate simulator engine.
"""
from src.array import SystolicArray
from src.workloads import skew_matrices, im2col, col2im
import numpy as np

class Simulator:
    def __init__(self, array_rows, array_cols, dataflow="OS"):
        self.array_rows = array_rows
        self.array_cols = array_cols
        self.dataflow = dataflow
        self.array = SystolicArray(array_rows, array_cols, dataflow=dataflow)
        
        self.total_cycles = 0
        self.history = []  # To store activity per cycle
        
    def run(self, A, B):
        """
        Run the simulation for matrix multiplication C = A x B.
        Handles tiling if A and B are larger than the array.
        """
        if self.dataflow == "OS":
            return self._run_os(A, B)
        elif self.dataflow == "WS":
            return self._run_ws(A, B)
        elif self.dataflow == "RS":
            return self._run_rs(A, B)
        else:
            raise ValueError(f"Unknown dataflow: {self.dataflow}")
            
    def _run_os(self, A, B):
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

    def _run_ws(self, A, B):
        M, K = A.shape
        K_b, N = B.shape
        assert K == K_b, "Inner dimensions must match"
        
        C = np.zeros((M, N))
        
        # Tile execution
        # WS: pre-load weight tile of shape (array_rows, array_cols)
        # So we tile K by array_rows, and N by array_cols.
        # We tile M by array_rows.
        for m in range(0, M, self.array_rows):
            for n in range(0, N, self.array_cols):
                for k in range(0, K, self.array_rows):
                    # Get slices
                    A_slice = A[m:m+self.array_rows, k:k+self.array_rows]
                    B_slice = B[k:k+self.array_rows, n:n+self.array_cols]
                    
                    # Pad
                    A_padded = np.zeros((self.array_rows, self.array_rows))
                    B_padded = np.zeros((self.array_rows, self.array_cols))
                    
                    actual_m = A_slice.shape[0]
                    actual_k = A_slice.shape[1]
                    actual_n = B_slice.shape[1]
                    
                    A_padded[:actual_m, :actual_k] = A_slice
                    B_padded[:actual_k, :actual_n] = B_slice
                    
                    # Reset array and load weights
                    self.array.reset()
                    self.array.load_weights(B_padded)
                    
                    # Run cycles
                    num_cycles = 2 * self.array_rows + self.array_cols - 2
                    C_tile = np.zeros((self.array_rows, self.array_cols))
                    
                    for cycle in range(num_cycles):
                        # Construct activation inputs (streamed from left)
                        act_in = []
                        for r in range(self.array_rows):
                            m_idx = cycle - r
                            if 0 <= m_idx < self.array_rows:
                                act_in.append(A_padded[m_idx, r])
                            else:
                                act_in.append(0.0)
                                
                        # Partial sums from top are always 0s
                        wt_in = [0.0] * self.array_cols
                        
                        self.array.step(act_in, wt_in)
                        self.total_cycles += 1
                        
                        # Record activity
                        self.history.append(self.array.get_pe_activity())
                        
                        # Read outputs from bottom row
                        for c in range(self.array_cols):
                            m_idx = cycle - c - (self.array_rows - 1)
                            if 0 <= m_idx < self.array_rows:
                                C_tile[m_idx, c] = self.array.grid[self.array_rows - 1][c].partial_sum_out
                                
                    # Accumulate partial products
                    C[m:m+actual_m, n:n+actual_n] += C_tile[:actual_m, :actual_n]
                    
        return C

    def _run_rs(self, A, B):
        M, K = A.shape
        K_b, N = B.shape
        assert K == K_b, "Inner dimensions must match"
        
        C = np.zeros((M, N))
        
        # Tile execution
        # RS: pre-load activation tile of shape (array_rows, array_cols)
        # So we tile M by array_rows, and K by array_cols.
        # We tile N by array_cols.
        for m in range(0, M, self.array_rows):
            for k in range(0, K, self.array_cols):
                for n in range(0, N, self.array_cols):
                    # Get slices
                    A_slice = A[m:m+self.array_rows, k:k+self.array_cols]
                    B_slice = B[k:k+self.array_cols, n:n+self.array_cols]
                    
                    # Pad
                    A_padded = np.zeros((self.array_rows, self.array_cols))
                    B_padded = np.zeros((self.array_cols, self.array_cols))
                    
                    actual_m = A_slice.shape[0]
                    actual_k = A_slice.shape[1]
                    actual_n = B_slice.shape[1]
                    
                    A_padded[:actual_m, :actual_k] = A_slice
                    B_padded[:actual_k, :actual_n] = B_slice
                    
                    # Reset array and load activations
                    self.array.reset()
                    self.array.load_activations(A_padded)
                    
                    # Run cycles
                    num_cycles = 2 * self.array_cols + self.array_rows - 2
                    C_tile = np.zeros((self.array_rows, self.array_cols))
                    
                    for cycle in range(num_cycles):
                        # Construct weight inputs (streamed from top)
                        wt_in = []
                        for c in range(self.array_cols):
                            k_idx = cycle - c
                            if 0 <= k_idx < self.array_cols:
                                wt_in.append(B_padded[c, k_idx])
                            else:
                                wt_in.append(0.0)
                                
                        # Partial sums from left are always 0s
                        act_in = [0.0] * self.array_rows
                        
                        self.array.step(act_in, wt_in)
                        self.total_cycles += 1
                        
                        # Record activity
                        self.history.append(self.array.get_pe_activity())
                        
                        # Read outputs from right column
                        for r in range(self.array_rows):
                            n_idx = cycle - r - (self.array_cols - 1)
                            if 0 <= n_idx < self.array_cols:
                                C_tile[r, n_idx] = self.array.grid[r][self.array_cols - 1].activation_out
                                
                    # Accumulate partial products
                    C[m:m+actual_m, n:n+actual_n] += C_tile[:actual_m, :actual_n]
                    
        return C

    def run_conv2d(self, image, filters):
        """
        Run simulation for 2D convolution.
        
        Args:
            image: numpy array of shape (batch, in_channels, height, width)
            filters: numpy array of shape (out_channels, in_channels, kernel_size, kernel_size)
            
        Returns:
            output: numpy array of shape (batch, out_channels, out_h, out_w)
        """
        # Lower convolution to GEMM
        A, B, out_shape = im2col(image, filters)
        
        # Run GEMM on the simulator (C = A x B)
        C = self.run(A, B)
        
        # Reconstruct back to 4D tensor
        output = col2im(C, out_shape)
        
        return output
