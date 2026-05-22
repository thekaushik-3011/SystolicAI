# SystolicAI

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

**SystolicAI** is a configurable AI accelerator simulator that models the execution of neural network workloads on a systolic-array-based architecture, similar to modern AI accelerators such as Google TPU architectures.

This project was built for educational, research, and hardware exploration purposes, allowing users to experiment with different accelerator configurations and analyze architectural tradeoffs in AI hardware systems.

## 🎯 Overview

Modern AI workloads rely heavily on matrix multiplication and convolution operations. Traditional CPUs are inefficient for these workloads due to limited parallelism and memory bottlenecks. Dedicated AI accelerators solve this using systolic arrays, parallel MAC units, optimized dataflow, and local memory reuse.

SystolicAI aims to simulate these architectural principles in software. Instead of running computations directly on CPU/GPU APIs, SystolicAI models:
- Compute pipelines
- Systolic data propagation
- Execution latency
- Hardware utilization
- Memory bandwidth effects

## ✨ Key Features (Phase 1)

1. **Processing Element (PE)**: Cycle-accurate MAC operations, local accumulation, and data forwarding.
2. **Systolic Array Grid**: Configurable PE grids (e.g., 2x2, 4x4, 8x8) supporting synchronized data propagation.
3. **Matrix Multiplication Engine**: Tiled execution and streaming inputs for $C = A \times B$.
4. **Cycle-Accurate Simulation**: Advances one clock cycle at a time, tracking PE activity and compute latency.
5. **Performance Metrics**: Reports execution cycles, throughput (MACs/cycle), and PE utilization.
6. **Visualization Engine**: Generates heatmap visualizations of array activity and line plots for utilization over time.

## 🏗️ System Architecture

```mermaid
graph TD
    A[Input Workload] --> B[Scheduler]
    B --> C[Dataflow Controller]
    C --> D[Systolic Array Simulator]
    D --> E[Cycle Engine]
    E --> F[Metrics Collector]
    F --> G[Visualization & Analysis]
```

### Internal Modules
- `pe.py`: Processing Element logic
- `array.py`: Systolic array management
- `simulator.py`: Cycle execution engine
- `metrics.py`: Performance analysis
- `visualize.py`: Heatmaps and graphs
- `workloads.py`: Matrix workloads

## 🚀 Setup & Installation

1. Clone the repository and navigate to the project directory.
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

Run the provided matrix multiplication example script to simulate execution, calculate metrics, and generate visualizations.

```bash
python examples/run_matmul.py
```

**Example Output Metrics:**
```
Array Size: 4x4
Matrix Size: 8x8 * 8x8
Total MACs: 512
Total Cycles: 60
PE Utilization (%): 53.33
Throughput (MACs/cycle): 8.53
```

This script will automatically create an `assets/` folder containing performance plots and heatmaps.

## 🧪 Testing

The simulator is verified against standard NumPy operations. Run the test suite using `pytest`:

```bash
pytest tests/
```

## 🗺️ Roadmap

- **Phase 1 (Current):** Basic systolic simulator, output stationary matmul, metrics, and visualization.
- **Phase 2:** CNN convolution support, multiple dataflows (weight stationary, row stationary).
- **Phase 3:** Sparse matrix acceleration, quantization support, transformer attention kernels.
- **Phase 4:** RTL backend, FPGA mapping, RISC-V custom instruction integration.
