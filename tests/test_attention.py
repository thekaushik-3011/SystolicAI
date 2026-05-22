"""
Unit tests for Self-Attention and Sparsity.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.workloads import (
    prune_unstructured, 
    prune_2to4, 
    generate_attention_workload
)
from src.simulator import Simulator
from src.metrics import compute_metrics

def test_sparsity_pruning():
    # Test unstructured sparsity
    np.random.seed(42)
    A = np.random.uniform(1.0, 10.0, size=(10, 10))
    A_sparse = prune_unstructured(A, 0.5)
    zeros = np.sum(A_sparse == 0.0)
    # Check that about 50% are zeroes
    assert 40 <= zeros <= 60
    
    # Test 2:4 structured sparsity
    B = np.random.uniform(1.0, 10.0, size=(8, 12))
    B_sparse = prune_2to4(B)
    for r in range(8):
        for c in range(0, 12, 4):
            group = B_sparse[r, c:c+4]
            # Must have exactly 2 zeroes in each block of 4
            assert np.sum(group == 0.0) == 2

@pytest.mark.parametrize("sparsity_type", ["unstructured", "2to4"])
def test_sparse_simulator_matmul(sparsity_type):
    # Test that running with sparse inputs counts skipped ops and saves dynamic power
    sim = Simulator(array_rows=4, array_cols=4, dataflow="OS")
    np.random.seed(42)
    A = np.random.uniform(1.0, 10.0, size=(8, 8))
    B = np.random.uniform(1.0, 10.0, size=(8, 8))
    
    if sparsity_type == "unstructured":
        A_pruned = prune_unstructured(A, 0.5)
    else:
        A_pruned = prune_2to4(A)
        
    C_sim = sim.run(A_pruned, B)
    metrics = compute_metrics(sim, A.shape, B.shape)
    
    # Verify skipped ops and energy savings
    assert sim.skipped_ops > 0
    assert metrics["Zero-skipped MACs"] > 0
    assert metrics["Dynamic Energy Saved (%)"] > 0.0
    
    # Verify mathematical correctness remains unaffected
    C_ref = np.matmul(A_pruned, B)
    np.testing.assert_allclose(C_sim, C_ref, rtol=1e-6)

@pytest.mark.parametrize("dataflow", ["OS", "WS", "RS"])
def test_attention_workload(dataflow):
    # Test running a full Self-Attention layer sequence on the simulator
    sim = Simulator(array_rows=4, array_cols=4, dataflow=dataflow)
    
    batch = 1
    seq_len = 4
    num_heads = 2
    head_dim = 4
    d_model = num_heads * head_dim
    
    Q_in, K_in, V_in, W_q, W_k, W_v = generate_attention_workload(
        batch, seq_len, num_heads, head_dim, seed=42
    )
    
    # Run simulation
    O_sim = sim.run_attention(Q_in, K_in, V_in, W_q, W_k, W_v)
    
    # NumPy Direct Reference implementation
    Q_flat = Q_in.reshape(-1, d_model)
    K_flat = K_in.reshape(-1, d_model)
    V_flat = V_in.reshape(-1, d_model)
    
    Q_ref = np.matmul(Q_flat, W_q).reshape(batch, seq_len, d_model)
    K_ref = np.matmul(K_flat, W_k).reshape(batch, seq_len, d_model)
    V_ref = np.matmul(V_flat, W_v).reshape(batch, seq_len, d_model)
    
    scale_factor = 1.0 / np.sqrt(d_model)
    O_ref = np.zeros((batch, seq_len, d_model))
    for b in range(batch):
        Q_b = Q_ref[b]
        K_b = K_ref[b]
        V_b = V_ref[b]
        
        S_b = np.matmul(Q_b, K_b.T) * scale_factor
        S_max = np.max(S_b, axis=-1, keepdims=True)
        exp_S = np.exp(S_b - S_max)
        P_b = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
        
        O_b = np.matmul(P_b, V_b)
        O_ref[b] = O_b
        
    np.testing.assert_allclose(O_sim, O_ref, rtol=1e-6)
