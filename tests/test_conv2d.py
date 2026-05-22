"""
Tests for 2D Convolution on the Systolic Array Simulator.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.workloads import generate_conv2d, reference_conv2d
from src.simulator import Simulator

@pytest.mark.parametrize("dataflow", ["OS", "WS", "RS"])
@pytest.mark.parametrize("array_size", [2, 4])
@pytest.mark.parametrize("batch, in_channels, height, width, out_channels, kernel_size", [
    (1, 1, 4, 4, 1, 3), # Small simple shape
    (2, 3, 5, 5, 2, 3), # Multi-channel, multi-batch, multi-filter
    (1, 2, 6, 6, 3, 2), # Tiled and padded cases
])
def test_conv2d_correctness(dataflow, array_size, batch, in_channels, height, width, out_channels, kernel_size):
    image, filters = generate_conv2d(batch, in_channels, height, width, out_channels, kernel_size, seed=42)
    
    sim = Simulator(array_rows=array_size, array_cols=array_size, dataflow=dataflow)
    conv_sim = sim.run_conv2d(image, filters)
    
    conv_ref = reference_conv2d(image, filters)
    
    np.testing.assert_allclose(conv_sim, conv_ref, err_msg=f"Failed Conv2D for dataflow {dataflow}, array {array_size}x{array_size}")
