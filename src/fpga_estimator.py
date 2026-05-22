"""
FPGA Resource Utilization Estimator.
"""
import math

class FPGAEstimator:
    def __init__(self, rows, cols, dataflow="OS", precision="FP32"):
        self.rows = rows
        self.cols = cols
        self.dataflow = dataflow
        self.precision = precision
        
        # Target Reference Device: AMD Xilinx Zynq UltraScale+ ZU3EG
        self.device_limits = {
            "DSPs": 360,
            "LUTs": 70560,
            "FFs": 141120,
            "BRAMs": 216
        }

    def estimate_resources(self):
        """
        Calculate FPGA resource usage based on array configuration.
        """
        # Determine widths
        input_width = 8 if self.precision == "INT8" else 32
        accum_width = 32
        
        # 1. DSP Blocks Estimation
        # FP32 MAC requires 4 DSP blocks (1 multiplier + 1 adder split)
        # INT8 MAC uses 1 DSP block (can pack easily)
        dsp_multiplier = 4 if self.precision == "FP32" else 1
        dsps = self.rows * self.cols * dsp_multiplier
        
        # 2. Flip-Flops (FFs) Estimation
        # Accumulator (accum_width) + Input regs (2 * input_width) + control registers (12 FFs per PE)
        ffs_per_pe = accum_width + 2 * input_width + 12
        pe_ffs = self.rows * self.cols * ffs_per_pe
        # Add shift registers for boundary conditions and top level controller
        top_ffs = (self.rows + self.cols) * input_width + 120
        ffs = pe_ffs + top_ffs
        
        # 3. Look-Up Tables (LUTs) Estimation
        # Base adder logic (accum_width) + multiplexing (50) + control logic (30)
        lut_base = accum_width + 80
        # Quantization clamping logic overhead for INT8
        clamping_overhead = 80 if self.precision == "INT8" else 0
        luts_per_pe = lut_base + clamping_overhead
        
        pe_luts = self.rows * self.cols * luts_per_pe
        # Add control FSM and routing logic
        top_luts = self.rows * self.cols * 15 + 250
        luts = pe_luts + top_luts
        
        # 4. Block RAMs (BRAMs) Estimation
        # Assume 1024-deep input SRAM buffers on rows and cols
        # A single Xilinx BRAM36K = 36,864 bits
        act_mem_bits = self.rows * 1024 * input_width
        wt_mem_bits = self.cols * 1024 * input_width
        psum_mem_bits = self.cols * 1024 * accum_width
        total_mem_bits = act_mem_bits + wt_mem_bits + psum_mem_bits
        
        brams = math.ceil(total_mem_bits / 36864)
        
        # Calculate percentage utilization relative to target device ZU3EG
        results = {
            "DSP Blocks Used": dsps,
            "DSP Utilization (%)": round((dsps / self.device_limits["DSPs"]) * 100, 2),
            "LUTs Used": luts,
            "LUT Utilization (%)": round((luts / self.device_limits["LUTs"]) * 100, 2),
            "FFs Used": ffs,
            "FF Utilization (%)": round((ffs / self.device_limits["FFs"]) * 100, 2),
            "BRAMs Used": brams,
            "BRAM Utilization (%)": round((brams / self.device_limits["BRAMs"]) * 100, 2),
            "Ref Device": "Xilinx Zynq UltraScale+ ZU3EG"
        }
        
        return results
