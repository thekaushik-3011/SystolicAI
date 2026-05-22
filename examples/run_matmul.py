"""
Example script to run matrix multiplication on the Systolic Array Simulator.
"""
import os
import sys

# Ensure src is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workloads import generate_matrices
from src.simulator import Simulator
from src.metrics import compute_metrics
from src.visualize import plot_utilization_over_time, plot_heatmap
import numpy as np

def main():
    # Configuration
    ARRAY_SIZE = 4
    MATRIX_SIZE = 8
    
    print(f"--- SystolicAI: Matrix Multiplication Example ---")
    print(f"Array Size: {ARRAY_SIZE}x{ARRAY_SIZE}")
    print(f"Matrix Size: {MATRIX_SIZE}x{MATRIX_SIZE}")
    
    # Generate data
    A, B = generate_matrices(MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, seed=42)
    print("\nInput Matrix A (Top left 4x4):")
    print(A[:4, :4])
    
    # Initialize simulator
    sim = Simulator(array_rows=ARRAY_SIZE, array_cols=ARRAY_SIZE)
    
    # Run
    print("\nRunning simulation...")
    C_sim = sim.run(A, B)
    
    # Validate against numpy
    C_np = np.matmul(A, B)
    is_correct = np.allclose(C_sim, C_np)
    print(f"\nSimulation Correctness: {'PASSED' if is_correct else 'FAILED'}")
    
    # Metrics
    metrics = compute_metrics(sim, A.shape, B.shape)
    print("\nPerformance Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    # Visualizations
    os.makedirs('assets', exist_ok=True)
    plot_utilization_over_time(sim.history, save_path='assets/utilization.png')
    print("\nSaved utilization plot to assets/utilization.png")
    
    # Save a heatmap from a busy cycle (around the middle of the first tile execution)
    # Total cycles for one tile = array_rows + array_cols + K - 2 + 1. For 4x4 with K=8, that's 4+4+8-2+1 = 15.
    # The first 15 cycles in history correspond to the first tile.
    if len(sim.history) > 8:
        busy_cycle = 8
        plot_heatmap(sim.history[busy_cycle], cycle=busy_cycle, save_path=f'assets/heatmap_cycle_{busy_cycle}.png')
        print(f"Saved PE activity heatmap to assets/heatmap_cycle_{busy_cycle}.png")

if __name__ == "__main__":
    main()
