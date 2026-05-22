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

@pytest.mark.parametrize("dataflow", ["OS", "WS", "RS"])
@pytest.mark.parametrize("array_size, matrix_size", [
    (2, 2),
    (4, 4),
    (4, 8),  # Tiled execution
    (8, 4),  # Smaller matrix than array
])
def test_matmul_correctness(dataflow, array_size, matrix_size):
    A, B = generate_matrices(matrix_size, matrix_size, matrix_size, seed=42)
    
    sim = Simulator(array_rows=array_size, array_cols=array_size, dataflow=dataflow)
    C_sim = sim.run(A, B)
    
    C_np = np.matmul(A, B)
    
    np.testing.assert_allclose(C_sim, C_np, err_msg=f"Failed for dataflow {dataflow}, array {array_size}x{array_size} and matrix {matrix_size}x{matrix_size}")

@pytest.mark.parametrize("dataflow", ["OS", "WS", "RS"])
@pytest.mark.parametrize("array_size, M, K, N", [
    (4, 6, 8, 10),
    (4, 3, 5, 2),
    (2, 5, 2, 7),
])
def test_matmul_non_square(dataflow, array_size, M, K, N):
    A, B = generate_matrices(M, K, N, seed=42)
    
    sim = Simulator(array_rows=array_size, array_cols=array_size, dataflow=dataflow)
    C_sim = sim.run(A, B)
    
    C_np = np.matmul(A, B)
    
    np.testing.assert_allclose(C_sim, C_np, err_msg=f"Failed for dataflow {dataflow}, array {array_size}x{array_size} and shape ({M}x{K}) * ({K}x{N})")
