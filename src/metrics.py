"""
Performance metrics collection.
"""

def compute_metrics(simulator, matrix_shape_A, matrix_shape_B, total_macs=None):
    """
    Calculate performance metrics from a completed simulation.
    """
    M, K = matrix_shape_A
    K_b, N = matrix_shape_B
    
    if total_macs is None:
        total_macs = M * N * K
    total_cycles = simulator.total_cycles
    
    # Calculate utilization
    active_pes = 0
    total_pes_over_time = total_cycles * simulator.array_rows * simulator.array_cols
    
    for activity in simulator.history:
        for row in activity:
            active_pes += sum(row)
            
    utilization = (active_pes / total_pes_over_time) * 100 if total_pes_over_time > 0 else 0
    throughput = (total_macs / total_cycles) if total_cycles > 0 else 0
    
    # Calculate energy saved by skipping MACs on zero inputs
    dynamic_energy_saved = (simulator.skipped_ops / total_pes_over_time) * 100 if total_pes_over_time > 0 else 0
    
    return {
        "Array Size": f"{simulator.array_rows}x{simulator.array_cols}",
        "Matrix Size": f"{M}x{K} * {K_b}x{N}",
        "Total MACs": total_macs,
        "Total Cycles": total_cycles,
        "PE Utilization (%)": round(utilization, 2),
        "Throughput (MACs/cycle)": round(throughput, 2),
        "Zero-skipped MACs": simulator.skipped_ops,
        "Dynamic Energy Saved (%)": round(dynamic_energy_saved, 2)
    }
