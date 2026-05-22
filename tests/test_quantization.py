import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.workloads import (
    generate_matrices, 
    get_quantization_params, 
    quantize_matrix, 
    dequantize_matrix
)
from src.simulator import Simulator

def test_quantization_helpers():
    # Test matrix quantization & dequantization roundtrip
    np.random.seed(42)
    A = np.random.uniform(-10.0, 10.0, size=(10, 10))
    scale, zp = get_quantization_params(A)
    
    A_quant = quantize_matrix(A, scale, zp)
    # Check that quantized values are integers within [-128, 127]
    assert np.all(A_quant >= -128)
    assert np.all(A_quant <= 127)
    assert np.all(np.isclose(A_quant, np.round(A_quant)))
    
    A_dequant = dequantize_matrix(A_quant, scale, zp)
    # Average absolute error should be small
    mae = np.mean(np.abs(A - A_dequant))
    assert mae < 0.1

@pytest.mark.parametrize("dataflow", ["OS", "WS", "RS"])
def test_int8_simulator_matmul(dataflow):
    # Test matrix multiplication on array simulator in INT8 mode
    sim = Simulator(array_rows=4, array_cols=4, dataflow=dataflow, precision="INT8")
    
    # Generate matrices with values in [1, 10]
    A, B = generate_matrices(8, 8, 8, seed=42)
    C_sim = sim.run(A, B)
    C_ref = np.matmul(A, B)
    
    # Average absolute error (quantization noise) should be small
    mae = np.mean(np.abs(C_sim - C_ref))
    assert mae < 1.0
