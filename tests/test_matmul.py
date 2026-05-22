"""
Tests for the Systolic Array Simulator.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.workloads import generate_matrices
from src.simulator import Simulator

@pytest.mark.parametrize("array_size, matrix_size", [
    (2, 2),
    (4, 4),
    (4, 8),  # Tiled execution
    (8, 4),  # Smaller matrix than array
])
def test_matmul_correctness(array_size, matrix_size):
    A, B = generate_matrices(matrix_size, matrix_size, matrix_size, seed=42)
    
    sim = Simulator(array_rows=array_size, array_cols=array_size)
    C_sim = sim.run(A, B)
    
    C_np = np.matmul(A, B)
    
    np.testing.assert_allclose(C_sim, C_np, err_msg=f"Failed for array {array_size}x{array_size} and matrix {matrix_size}x{matrix_size}")
