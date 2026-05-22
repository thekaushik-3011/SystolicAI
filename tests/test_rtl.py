"""
Unit tests for Verilog RTL Generator.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.rtl_generator import RTLGenerator

@pytest.mark.parametrize("dataflow", ["OS", "WS", "RS"])
@pytest.mark.parametrize("precision", ["FP32", "INT8"])
def test_rtl_generation(dataflow, precision):
    generator = RTLGenerator(rows=4, cols=4, dataflow=dataflow, precision=precision)
    
    # Generate modules
    pe_verilog = generator.generate_pe()
    array_verilog = generator.generate_array()
    complete_verilog = generator.generate_complete_system()
    
    # Assert structural content
    assert "module processing_element" in pe_verilog
    assert "module systolic_array" in array_verilog
    assert "systolic_array" in complete_verilog
    assert "processing_element" in complete_verilog
    
    # Dataflow specifics
    if dataflow == "OS":
        assert "accumulator" in pe_verilog
    elif dataflow == "WS":
        assert "weight_reg" in pe_verilog
        assert "weight_load" in array_verilog
    else:  # RS
        assert "act_reg" in pe_verilog
        assert "act_load" in array_verilog
        
    # Bitwidths
    if precision == "INT8":
        assert "INPUT_WIDTH = 8" in pe_verilog
    else:
        assert "INPUT_WIDTH = 32" in pe_verilog
