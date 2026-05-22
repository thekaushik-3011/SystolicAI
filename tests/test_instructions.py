"""
Unit tests for Custom RISC-V ISA Simulator.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.instruction_set import RISCVExtensionSimulator

def test_riscv_base_instructions():
    emu = RISCVExtensionSimulator()
    
    # Run simple program
    program = """
    li x1, 42
    li x2, 10
    add x3, x1, x2
    sub x4, x1, x2
    """
    emu.execute_program(program)
    
    # Check register values
    assert emu.registers["x1"] == 42
    assert emu.registers["x2"] == 10
    assert emu.registers["x3"] == 52
    assert emu.registers["x4"] == 32
    assert emu.registers["x0"] == 0  # Hardwired to 0

def test_riscv_systolic_instructions():
    emu = RISCVExtensionSimulator()
    
    # Setup matrices in simulated host memory
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], dtype=np.float32)
    B = np.array([
        [5.0, 6.0],
        [7.0, 8.0]
    ], dtype=np.float32)
    
    # Load A at address 1000, B at 2000
    emu.load_memory(1000, A)
    emu.load_memory(2000, B)
    
    # Run custom assembly
    program = """
    # 1. Config systolic accelerator (2x2 grid, Output Stationary, FP32)
    syst_cfg 2, 2, OS, FP32
    
    # 2. Set pointers in registers
    li x1, 1000  # Address of A
    li x2, 2000  # Address of B
    li x3, 3000  # Address of output C
    
    # 3. Dispatch GEMM to accelerator
    syst_exec x3, x1, x2, 2, 2, 2
    """
    emu.execute_program(program)
    
    # Verify C matches standard matrix multiplication
    C_sim = emu.read_memory(3000)
    assert C_sim is not None
    C_ref = np.matmul(A, B)
    np.testing.assert_allclose(C_sim, C_ref, rtol=1e-6)
