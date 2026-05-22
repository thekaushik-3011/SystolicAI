import streamlit as st
import numpy as np

# Ensure src is accessible
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.workloads import generate_matrices, generate_conv2d, im2col, reference_conv2d
from src.simulator import Simulator
from src.metrics import compute_metrics
from src.visualize import plot_heatmap, plot_utilization_over_time

def main():
    st.set_page_config(
        page_title="SystolicAI — AI Accelerator Simulator",
        page_icon="🖥️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for premium styling
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stMetric {
            background-color: #1f2937;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #374151;
            text-align: center;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #3b82f6;
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #9ca3af;
        }
        .subheader {
            color: #60a5fa;
            border-bottom: 2px solid #1f2937;
            padding-bottom: 5px;
            margin-top: 25px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🖥️ SystolicAI: Hardware Accelerator Simulator")
    st.markdown(
        "A cycle-accurate architectural simulator modeling neural network workload execution "
        "on spatial systolic-array architectures. Analyze processing element (PE) utilization, dataflow "
        "propagation behaviors, and performance metrics."
    )
    st.divider()

    # Sidebar configuration
    st.sidebar.image("https://img.icons8.com/nolan/96/cpu.png", width=80)
    st.sidebar.title("Configuration")
    
    st.sidebar.subheader("1. Hardware Design")
    array_size = st.sidebar.slider("Array Dimensions (N x N)", min_value=2, max_value=16, value=4, step=2)
    
    dataflow = st.sidebar.selectbox(
        "Dataflow Architecture",
        options=["OS", "WS", "RS"],
        format_func=lambda x: {
            "OS": "Output Stationary (OS) — TPUs / Local Accumulation",
            "WS": "Weight Stationary (WS) — Pinned Weights / Vertical PSum",
            "RS": "Row Stationary (RS) — Pinned Activations / Horizontal PSum"
        }[x],
        help="OS: Accumulates partial sums locally in PEs. WS: Weights are stationary in PEs. RS: Activations are stationary in PEs."
    )
    
    st.sidebar.subheader("2. Workload Mapping")
    workload_type = st.sidebar.radio(
        "Workload Type",
        options=["GEMM Matrix Multiplication", "Conv2D Convolution"]
    )
    
    # Workload parameters
    if workload_type == "GEMM Matrix Multiplication":
        st.sidebar.markdown("**Matrix Shapes (A x B)**")
        m_dim = st.sidebar.slider("M (Rows of A)", min_value=2, max_value=32, value=8, step=2)
        k_dim = st.sidebar.slider("K (Cols of A / Rows of B)", min_value=2, max_value=32, value=8, step=2)
        n_dim = st.sidebar.slider("N (Cols of B)", min_value=2, max_value=32, value=8, step=2)
    else:
        st.sidebar.markdown("**Conv2D Parameters**")
        batch = st.sidebar.slider("Batch Size", min_value=1, max_value=4, value=1)
        in_channels = st.sidebar.slider("Input Channels", min_value=1, max_value=8, value=3)
        height = st.sidebar.slider("Input Height", min_value=4, max_value=16, value=8)
        width = st.sidebar.slider("Input Width", min_value=4, max_value=16, value=8)
        out_channels = st.sidebar.slider("Output Channels (Filters)", min_value=1, max_value=8, value=4)
        kernel_size = st.sidebar.slider("Kernel Size", min_value=2, max_value=5, value=3)

    st.sidebar.markdown("---")
    run_sim = st.sidebar.button("Run Simulation 🚀", type="primary", use_container_width=True)

    if run_sim:
        with st.spinner("Simulating cycle-accurate execution..."):
            sim = Simulator(array_rows=array_size, array_cols=array_size, dataflow=dataflow)
            
            if workload_type == "GEMM Matrix Multiplication":
                A, B = generate_matrices(m_dim, k_dim, n_dim, seed=42)
                C_sim = sim.run(A, B)
                shape_A, shape_B = A.shape, B.shape
                # Verification
                C_ref = np.matmul(A, B)
                is_correct = np.allclose(C_sim, C_ref)
            else:
                image, filters = generate_conv2d(batch, in_channels, height, width, out_channels, kernel_size, seed=42)
                # Run convolution
                conv_out = sim.run_conv2d(image, filters)
                # Get equivalent GEMM shapes for metrics
                A, B, _ = im2col(image, filters)
                shape_A, shape_B = A.shape, B.shape
                # Verification
                conv_ref = reference_conv2d(image, filters)
                is_correct = np.allclose(conv_out, conv_ref)
            
            # Compute Metrics
            metrics = compute_metrics(sim, shape_A, shape_B)
            
            # Store in session state to persist during slider scrubbing
            st.session_state['sim_history'] = sim.history
            st.session_state['metrics'] = metrics
            st.session_state['total_cycles'] = sim.total_cycles
            st.session_state['dataflow_mode'] = dataflow
            st.session_state['workload_mode'] = workload_type
            st.session_state['shape_A'] = shape_A
            st.session_state['shape_B'] = shape_B
            st.session_state['is_correct'] = is_correct
            st.session_state['simulation_run'] = True

    if st.session_state.get('simulation_run', False):
        # Display Metrics
        st.markdown("<h3 class='subheader'>Performance Analysis</h3>", unsafe_allow_html=True)
        m = st.session_state['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total MAC Operations", f"{m['Total MACs']:,}")
        with col2:
            st.metric("Execution Latency (Cycles)", f"{m['Total Cycles']:,}")
        with col3:
            st.metric("PE Grid Utilization", f"{m['PE Utilization (%)']}%")
        with col4:
            st.metric("Array Throughput", f"{m['Throughput (MACs/cycle)']} MACs/Cycle")
            
        # Mathematical Correctness Status Card
        is_correct = st.session_state.get('is_correct', False)
        if is_correct:
            st.success("✅ **Mathematical Correctness Verified!** The systolic array simulation matches the NumPy reference output within $1e-7$ numerical tolerance.")
        else:
            st.error("❌ **Verification Failed!** The systolic array simulation output differs from the NumPy reference computation.")
            
        # Summary description card
        st.markdown(f"""
        > [!NOTE]
        > **Dataflow Architecture:** `{st.session_state['dataflow_mode']}` | **Workload Model:** `{st.session_state['workload_mode']}`
        > - **Lowered GEMM Size:** A ({st.session_state['shape_A'][0]} x {st.session_state['shape_A'][1]}) multiplied by B ({st.session_state['shape_B'][0]} x {st.session_state['shape_B'][1]})
        > - **Hardware Dimensions:** {st.session_state['metrics']['Array Size']} Systolic Grid
        """)
        
        st.divider()
        
        # Visualizations
        st.markdown("<h3 class='subheader'>Cycle-Accurate Visualizations</h3>", unsafe_allow_html=True)
        
        col_vis1, col_vis2 = st.columns([1, 1])
        
        total_cycles = st.session_state['total_cycles']
        history = st.session_state['sim_history']
        
        with col_vis1:
            st.markdown("### PE Grid Activity Heatmap")
            st.markdown("Scrub the slider to inspect the spatial state of the processing elements at any cycle.")
            cycle_to_view = st.slider("Select Clock Cycle", min_value=0, max_value=total_cycles - 1, value=0)
            
            fig_heatmap = plot_heatmap(history[cycle_to_view], cycle=cycle_to_view, return_fig=True)
            st.pyplot(fig_heatmap)
            
        with col_vis2:
            st.markdown("### Utilization Over Time")
            st.markdown("Total active processing elements performing MAC computations across the cycles.")
            
            fig_util = plot_utilization_over_time(history, return_fig=True)
            st.pyplot(fig_util)
            
        # Explain Dataflow Mapping
        st.markdown("<h3 class='subheader'>Dataflow Mapping Details</h3>", unsafe_allow_html=True)
        
        df_mode = st.session_state['dataflow_mode']
        if df_mode == "OS":
            st.markdown("""
            **Output Stationary (OS) Dataflow Details:**
            - **Stationary Operand:** The partial sums of the output matrix $C$ are held locally in each processing element's accumulator.
            - **Input Stream A (Left):** Activation values are shifted into the array row-by-row, delayed by one cycle per row (horizontal skewing).
            - **Input Stream B (Top):** Weights are shifted down the columns, delayed by one cycle per column (vertical skewing).
            - **Execution Phase:** After all streaming inputs pass through, the final output matrix is read directly from the local registers of the array.
            """)
        elif df_mode == "WS":
            st.markdown("""
            **Weight Stationary (WS) Dataflow Details:**
            - **Stationary Operand:** Weights are pre-loaded into the processing elements and remain fixed during computation.
            - **Input Stream A (Left):** Activation values are shifted horizontally across rows.
            - **Input Stream B (Top):** Partial sums (initialized to 0) are shifted down columns, accumulating products at each PE.
            - **Execution Phase:** The final results are streamed out from the bottom row of the systolic array and collected cycle-by-cycle.
            """)
        elif df_mode == "RS":
            st.markdown("""
            **Row Stationary (RS) Dataflow Details:**
            - **Stationary Operand:** Activations are pre-loaded into the processing elements and remain fixed during computation.
            - **Input Stream A (Left):** Partial sums (initialized to 0) are shifted horizontally across rows, accumulating products at each PE.
            - **Input Stream B (Top):** Weights are shifted down columns.
            - **Execution Phase:** The final results are streamed out from the rightmost column of the systolic array and collected cycle-by-cycle.
            """)

if __name__ == "__main__":
    main()
