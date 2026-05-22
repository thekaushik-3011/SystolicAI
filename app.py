import streamlit as st
import numpy as np

# Ensure src is accessible
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.workloads import (
    generate_matrices, 
    generate_conv2d, 
    im2col, 
    col2im,
    reference_conv2d,
    prune_unstructured,
    prune_2to4,
    generate_attention_workload
)
from src.simulator import Simulator
from src.metrics import compute_metrics
from src.visualize import plot_heatmap, plot_utilization_over_time
from src.rtl_generator import RTLGenerator
from src.fpga_estimator import FPGAEstimator
from src.instruction_set import RISCVExtensionSimulator

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
        .code-box {
            background-color: #111827;
            border: 1px solid #374151;
            border-radius: 5px;
            padding: 15px;
            color: #10b981;
            font-family: 'Courier New', Courier, monospace;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🖥️ SystolicAI: Hardware Accelerator Simulator")
    st.markdown(
        "A cycle-accurate architectural simulator modeling neural network workload execution "
        "on spatial systolic-array architectures. Analyze processing element (PE) utilization, dataflow "
        "propagation behaviors, precision options, sparsity savings, custom instruction flows, and synthesizable RTL hardware."
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
    
    st.sidebar.subheader("2. Precision Settings")
    precision = st.sidebar.selectbox(
        "Compute Precision",
        options=["FP32", "INT8"],
        help="FP32: Floating point computation. INT8: Simulated 8-bit integer computation with symmetric scaling and clipping."
    )
    
    st.sidebar.subheader("3. Sparsity & Power Optimization")
    sparsity_mode = st.sidebar.selectbox(
        "Activation Sparsity Mode",
        options=["None", "Unstructured", "2:4 Structured"],
        help="None: Dense computation. Unstructured: Randomly zero out activations. 2:4 Structured: Zero out 2 of every 4 elements."
    )
    if sparsity_mode == "Unstructured":
        sparsity_ratio = st.sidebar.slider("Sparsity Ratio (Zero %)", min_value=0.0, max_value=0.9, value=0.5, step=0.1)
    else:
        sparsity_ratio = 0.0

    st.sidebar.subheader("4. Workload Mapping")
    workload_type = st.sidebar.radio(
        "Workload Type",
        options=["GEMM Matrix Multiplication", "Conv2D Convolution", "Transformer Self-Attention"]
    )
    
    # Workload parameters
    if workload_type == "GEMM Matrix Multiplication":
        st.sidebar.markdown("**Matrix Shapes (A x B)**")
        m_dim = st.sidebar.slider("M (Rows of A)", min_value=2, max_value=32, value=8, step=2)
        k_dim = st.sidebar.slider("K (Cols of A / Rows of B)", min_value=2, max_value=32, value=8, step=2)
        n_dim = st.sidebar.slider("N (Cols of B)", min_value=2, max_value=32, value=8, step=2)
    elif workload_type == "Conv2D Convolution":
        st.sidebar.markdown("**Conv2D Parameters**")
        batch = st.sidebar.slider("Batch Size", min_value=1, max_value=4, value=1)
        in_channels = st.sidebar.slider("Input Channels", min_value=1, max_value=8, value=3)
        height = st.sidebar.slider("Input Height", min_value=4, max_value=16, value=8)
        width = st.sidebar.slider("Input Width", min_value=4, max_value=16, value=8)
        out_channels = st.sidebar.slider("Output Channels (Filters)", min_value=1, max_value=8, value=4)
        kernel_size = st.sidebar.slider("Kernel Size", min_value=2, max_value=5, value=3)
    else:
        st.sidebar.markdown("**Self-Attention Parameters**")
        batch = st.sidebar.slider("Batch Size", min_value=1, max_value=4, value=1)
        seq_len = st.sidebar.slider("Sequence Length", min_value=2, max_value=16, value=4)
        num_heads = st.sidebar.slider("Number of Heads", min_value=1, max_value=4, value=2)
        head_dim = st.sidebar.slider("Head Dimension", min_value=2, max_value=16, value=4)

    st.sidebar.markdown("---")
    run_sim = st.sidebar.button("Run Simulation 🚀", type="primary", use_container_width=True)

    if run_sim:
        with st.spinner("Simulating cycle-accurate execution..."):
            sim = Simulator(array_rows=array_size, array_cols=array_size, dataflow=dataflow, precision=precision)
            
            if workload_type == "GEMM Matrix Multiplication":
                A, B = generate_matrices(m_dim, k_dim, n_dim, seed=42)
                # Apply sparsity to activations (Matrix A)
                if sparsity_mode == "Unstructured":
                    A = prune_unstructured(A, sparsity_ratio)
                elif sparsity_mode == "2:4 Structured":
                    A = prune_2to4(A)
                
                C_sim = sim.run(A, B)
                shape_A, shape_B = A.shape, B.shape
                # Verification
                C_ref = np.matmul(A, B)
                is_correct = np.allclose(C_sim, C_ref, atol=2.0 if precision == "INT8" else 1e-7)
                metrics = compute_metrics(sim, shape_A, shape_B)
                
            elif workload_type == "Conv2D Convolution":
                image, filters = generate_conv2d(batch, in_channels, height, width, out_channels, kernel_size, seed=42)
                # Lower to GEMM
                A, B, out_shape = im2col(image, filters)
                # Apply sparsity to activations (lower matrix A)
                if sparsity_mode == "Unstructured":
                    A = prune_unstructured(A, sparsity_ratio)
                elif sparsity_mode == "2:4 Structured":
                    A = prune_2to4(A)
                
                C_sim = sim.run(A, B)
                conv_out = col2im(C_sim, out_shape)
                shape_A, shape_B = A.shape, B.shape
                # Verification
                C_ref = np.matmul(A, B)
                conv_ref = col2im(C_ref, out_shape)
                is_correct = np.allclose(conv_out, conv_ref, atol=2.0 if precision == "INT8" else 1e-7)
                metrics = compute_metrics(sim, shape_A, shape_B)
                
            else:
                Q_in, K_in, V_in, W_q, W_k, W_v = generate_attention_workload(
                    batch, seq_len, num_heads, head_dim, seed=42
                )
                O_sim = sim.run_attention(Q_in, K_in, V_in, W_q, W_k, W_v)
                
                # Verification
                d_model = num_heads * head_dim
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
                    
                is_correct = np.allclose(O_sim, O_ref, atol=2.0 if precision == "INT8" else 1e-7)
                
                # Precompute total MACs
                proj_macs = 3 * (batch * seq_len * d_model * d_model)
                attn_macs = batch * seq_len * seq_len * d_model
                out_macs = batch * seq_len * d_model * seq_len
                total_macs = proj_macs + attn_macs + out_macs
                
                shape_A = (batch * seq_len, d_model)
                shape_B = (d_model, d_model)
                metrics = compute_metrics(sim, shape_A, shape_B, total_macs=total_macs)
            
            # Store in session state to persist during slider scrubbing
            st.session_state['sim_history'] = sim.history
            st.session_state['metrics'] = metrics
            st.session_state['total_cycles'] = sim.total_cycles
            st.session_state['dataflow_mode'] = dataflow
            st.session_state['workload_mode'] = workload_type
            st.session_state['precision_mode'] = precision
            st.session_state['array_size'] = array_size
            st.session_state['sparsity_config'] = f"{sparsity_mode} ({sparsity_ratio * 100:.0f}% zero)" if sparsity_mode == "Unstructured" else sparsity_mode
            st.session_state['shape_A'] = shape_A
            st.session_state['shape_B'] = shape_B
            st.session_state['is_correct'] = is_correct
            st.session_state['simulation_run'] = True

    if st.session_state.get('simulation_run', False):
        # Create Tabs for different analytical views
        tab_perf, tab_rtl, tab_riscv = st.tabs([
            "📊 Performance & Visualizations", 
            "🔌 RTL & FPGA Synthesis", 
            "💻 RISC-V Custom Assembly"
        ])
        
        with tab_perf:
            # Display Metrics
            st.markdown("<h3 class='subheader'>Performance Analysis</h3>", unsafe_allow_html=True)
            m = st.session_state['metrics']
            
            # Row 1 metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total MAC Operations", f"{m['Total MACs']:,}")
            with col2:
                st.metric("Execution Latency (Cycles)", f"{m['Total Cycles']:,}")
            with col3:
                st.metric("PE Grid Utilization", f"{m['PE Utilization (%)']}%")
            with col4:
                st.metric("Array Throughput", f"{m['Throughput (MACs/cycle)']} MACs/Cycle")
                
            # Row 2 metrics
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Zero-skipped MACs", f"{m['Zero-skipped MACs']:,}")
            with col6:
                st.metric("Dynamic Energy Saved", f"{m['Dynamic Energy Saved (%)']}%")
            with col7:
                st.metric("Compute Precision", f"{st.session_state['precision_mode']}")
            with col8:
                st.metric("Sparsity Setting", f"{st.session_state['sparsity_config']}")
                
            # Mathematical Correctness Status Card
            is_correct = st.session_state.get('is_correct', False)
            if is_correct:
                if st.session_state['precision_mode'] == "INT8":
                    st.success("✅ **INT8 Correctness Verified!** The systolic array simulation matches the NumPy reference output within quantization noise limits (absolute tolerance 2.0).")
                else:
                    st.success("✅ **FP32 Correctness Verified!** The systolic array simulation matches the NumPy reference output within $1e-7$ numerical tolerance.")
            else:
                st.error("❌ **Verification Failed!** The systolic array simulation output differs from the NumPy reference computation.")
                
            # Summary description card
            st.markdown(f"""
            > [!NOTE]
            > **Dataflow Architecture:** `{st.session_state['dataflow_mode']}` | **Workload Model:** `{st.session_state['workload_mode']}` | **Precision Mode:** `{st.session_state['precision_mode']}`
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
                
        with tab_rtl:
            st.markdown("<h3 class='subheader'>RTL Verilog Hardware Export & FPGA Estimation</h3>", unsafe_allow_html=True)
            
            # Generate estimations
            estimator = FPGAEstimator(
                rows=st.session_state['array_size'], 
                cols=st.session_state['array_size'], 
                dataflow=st.session_state['dataflow_mode'], 
                precision=st.session_state['precision_mode']
            )
            est = estimator.estimate_resources()
            
            st.markdown(f"#### FPGA Resource Report (Target: `{est['Ref Device']}`)")
            
            col_est1, col_est2, col_est3, col_est4 = st.columns(4)
            with col_est1:
                st.metric("DSP Blocks", f"{est['DSP Blocks Used']}", f"{est['DSP Utilization (%)']}% of Target")
            with col_est2:
                st.metric("Look-Up Tables (LUTs)", f"{est['LUTs Used']:,}", f"{est['LUT Utilization (%)']}% of Target")
            with col_est3:
                st.metric("Flip-Flops (FFs)", f"{est['FFs Used']:,}", f"{est['FF Utilization (%)']}% of Target")
            with col_est4:
                st.metric("Block RAMs (BRAMs)", f"{est['BRAMs Used']}", f"{est['BRAM Utilization (%)']}% of Target")
                
            st.divider()
            
            # Generate Verilog code
            generator = RTLGenerator(
                rows=st.session_state['array_size'], 
                cols=st.session_state['array_size'], 
                dataflow=st.session_state['dataflow_mode'], 
                precision=st.session_state['precision_mode']
            )
            verilog_code = generator.generate_complete_system()
            
            st.markdown("#### Generated Synthesizable Verilog Code")
            st.code(verilog_code, language="verilog")
            
            st.download_button(
                label="Download Verilog Hardware Package (.v) 📥",
                data=verilog_code,
                file_name=f"systolic_array_{st.session_state['array_size']}x{st.session_state['array_size']}_{st.session_state['dataflow_mode']}.v",
                mime="text/plain",
                use_container_width=True
            )
            
        with tab_riscv:
            st.markdown("<h3 class='subheader'>RISC-V Custom Assembly Extension Editor</h3>", unsafe_allow_html=True)
            
            # Select program example
            program_choice = st.selectbox(
                "Select Pre-written Assembly Program Template",
                options=[
                    "Template 1: Vector-Vector Matrix Multiply (High-level Dispatch)",
                    "Template 2: Manual Cycle Stepping (Low-level Control)"
                ]
            )
            
            # Map selection to assembly templates
            if "Template 1" in program_choice:
                default_asm = f"""# RISC-V High-Level Systolic Accelerator Program
# Preloads weights and executes a 2x2 matrix multiplication in OS mode

# 1. Config hardware array
syst_cfg 2, 2, OS, FP32

# 2. Load mock pointers to pointer registers
li x1, 1000  # Host memory address of Matrix A
li x2, 2000  # Host memory address of Matrix B
li x3, 3000  # Host memory address of Output C

# 3. Dispatch execution to Systolic Tensor Engine
syst_exec x3, x1, x2, 2, 2, 2
"""
            else:
                default_asm = f"""# RISC-V Low-Level Manual Clock Gating Program
# Sets cycle inputs and steps the clock cycles manually

# 1. Config array
syst_cfg 2, 2, OS, FP32

# 2. Pointers of step inputs
li x1, 100   # Activation cycle inputs
li x2, 200   # Weight cycle inputs

# 3. Manually step clocks to propagate inputs
syst_step x1, x2
syst_step x1, x2
syst_step x1, x2
"""
            
            asm_code = st.text_area("RISC-V Program Editor", value=default_asm, height=220)
            run_asm = st.button("Compile & Assemble Code ⚙️", type="primary", use_container_width=True)
            
            if run_asm:
                try:
                    # Pre-load dummy datasets in host simulator memory
                    emu = RISCVExtensionSimulator()
                    
                    # Array A & B mock data
                    A_mock = np.array([[1.5, 2.0], [3.0, 4.5]], dtype=np.float32)
                    B_mock = np.array([[5.0, 1.0], [2.0, 3.5]], dtype=np.float32)
                    emu.load_memory(1000, A_mock)
                    emu.load_memory(2000, B_mock)
                    
                    # Cycle inputs mock data
                    emu.load_memory(100, np.array([2.5, 3.0], dtype=np.float32))
                    emu.load_memory(200, np.array([1.0, 4.0], dtype=np.float32))
                    
                    # Run the program
                    trace = emu.execute_program(asm_code)
                    
                    st.success("🎉 **Assembly Compiled and Executed Successfully!**")
                    
                    st.markdown("#### Hardware Execution Trace Log")
                    # Construct tabular format for trace log
                    trace_table = []
                    for entry in trace:
                        trace_table.append({
                            "PC (Addr)": f"0x{entry['pc']:04x}",
                            "Instruction": entry["instruction"],
                            "Execution Status": entry["status"],
                            "Accelerator Event": entry.get("accelerator_event", "N/A")
                        })
                    st.table(trace_table)
                    
                    # Display output registers
                    st.markdown("#### Final CPU Registers State")
                    cols = st.columns(8)
                    reg_list = list(emu.registers.keys())
                    for idx, reg in enumerate(reg_list):
                        with cols[idx % 8]:
                            st.text(f"{reg}: {emu.registers[reg]}")
                            
                    # Display resulting memory output
                    C_out = emu.read_memory(3000)
                    if C_out is not None:
                        st.markdown("#### Memory Result Output Address: `3000` (C Matrix)")
                        st.dataframe(C_out)
                        
                except Exception as ex:
                    st.error(f"❌ **Assembly Compiler Error:** {str(ex)}")

if __name__ == "__main__":
    main()
