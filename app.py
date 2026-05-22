import streamlit as st
import numpy as np

# Ensure src is accessible
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.workloads import generate_matrices
from src.simulator import Simulator
from src.metrics import compute_metrics
from src.visualize import plot_heatmap, plot_utilization_over_time

def main():
    st.set_page_config(page_title="SystolicAI Dashboard", layout="wide")
    
    st.title("🖥️ SystolicAI Simulator Dashboard")
    st.markdown("Explore matrix multiplication workloads on a simulated systolic array architecture.")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    st.sidebar.subheader("Hardware")
    array_size = st.sidebar.slider("Array Size (N x N)", min_value=2, max_value=16, value=4, step=2)
    
    st.sidebar.subheader("Workload")
    matrix_size = st.sidebar.slider("Matrix Size (M x M)", min_value=2, max_value=32, value=8, step=2)

    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner(f"Simulating {matrix_size}x{matrix_size} Matrix Multiplication on {array_size}x{array_size} Array..."):
            # Generate Matrices
            A, B = generate_matrices(matrix_size, matrix_size, matrix_size, seed=42)
            
            # Run Simulation
            sim = Simulator(array_rows=array_size, array_cols=array_size)
            C_sim = sim.run(A, B)
            
            # Compute Metrics
            metrics = compute_metrics(sim, A.shape, B.shape)
            
            # Store in session state to persist during scrubbing
            st.session_state['sim_history'] = sim.history
            st.session_state['metrics'] = metrics
            st.session_state['total_cycles'] = sim.total_cycles
            st.session_state['simulation_run'] = True

    if st.session_state.get('simulation_run', False):
        st.divider()
        
        # Display Metrics
        st.subheader("Performance Metrics")
        m = st.session_state['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total MACs", m["Total MACs"])
        col2.metric("Total Cycles", m["Total Cycles"])
        col3.metric("PE Utilization", f'{m["PE Utilization (%)"]}%')
        col4.metric("Throughput", f'{m["Throughput (MACs/cycle)"]} MACs/cycle')
        
        st.divider()
        
        # Visualizations
        st.subheader("Cycle-Accurate Visualizations")
        
        col_vis1, col_vis2 = st.columns([1, 1])
        
        total_cycles = st.session_state['total_cycles']
        
        with col_vis1:
            st.markdown("### PE Activity Heatmap")
            st.markdown("Scrub the slider below to view the active Processing Elements at any given clock cycle.")
            cycle_to_view = st.slider("Select Clock Cycle", min_value=0, max_value=total_cycles - 1, value=0)
            
            # Get the history for the selected cycle
            history = st.session_state['sim_history']
            fig_heatmap = plot_heatmap(history[cycle_to_view], cycle=cycle_to_view, return_fig=True)
            st.pyplot(fig_heatmap)
            
        with col_vis2:
            st.markdown("### Utilization Over Time")
            st.markdown("The global number of active PEs across the entire matrix multiplication execution.")
            
            history = st.session_state['sim_history']
            fig_util = plot_utilization_over_time(history, return_fig=True)
            st.pyplot(fig_util)

if __name__ == "__main__":
    main()
