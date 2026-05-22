"""
Custom RISC-V ISA Extension Emulator for Systolic Accelerator.
"""
import numpy as np
from src.simulator import Simulator

class RISCVExtensionSimulator:
    def __init__(self):
        # Register file: x0 is hardwired to 0, x1-x31 are general purpose
        self.registers = {f"x{i}": 0 for i in range(32)}
        self.memory = {}  # Sparse memory mapping: address (int) -> numpy array or float
        
        # Internal accelerator state
        self.sim = None
        self.pc = 0
        self.execution_trace = []

    def load_memory(self, address, data):
        """
        Helper to pre-load matrix or vector data into simulated system memory.
        """
        self.memory[int(address)] = np.array(data, dtype=np.float32)

    def read_memory(self, address):
        """
        Read value or matrix from memory.
        """
        return self.memory.get(int(address), None)

    def parse_instruction(self, instr_str):
        """
        Parse instruction string into components.
        Removes comments and formats operands.
        """
        instr_str = instr_str.split("#")[0].strip()  # Remove comments
        if not instr_str:
            return None, []
            
        parts = instr_str.replace(",", " ").split()
        opcode = parts[0].lower()
        operands = parts[1:]
        return opcode, operands

    def execute_instruction(self, instr_str):
        """
        Execute a single assembly instruction.
        """
        opcode, operands = self.parse_instruction(instr_str)
        if not opcode:
            return
            
        trace_entry = {
            "pc": self.pc,
            "instruction": instr_str,
            "register_state": self.registers.copy(),
            "status": "Success"
        }
        
        try:
            # 1. Base ISA Instructions
            if opcode == "li":
                rd, imm = operands[0], int(operands[1])
                if rd != "x0":
                    self.registers[rd] = imm
                    
            elif opcode == "add":
                rd, rs1, rs2 = operands[0], operands[1], operands[2]
                if rd != "x0":
                    self.registers[rd] = self.registers[rs1] + self.registers[rs2]
                    
            elif opcode == "sub":
                rd, rs1, rs2 = operands[0], operands[1], operands[2]
                if rd != "x0":
                    self.registers[rd] = self.registers[rs1] - self.registers[rs2]

            # 2. Custom Systolic Accelerator Extensions
            elif opcode == "syst_cfg":
                # syst_cfg rows, cols, dataflow, precision
                rows = int(operands[0])
                cols = int(operands[1])
                dataflow = operands[2].upper()
                precision = operands[3].upper() if len(operands) > 3 else "FP32"
                
                self.sim = Simulator(array_rows=rows, array_cols=cols, dataflow=dataflow, precision=precision)
                trace_entry["accelerator_event"] = f"Initialized {rows}x{cols} {dataflow} Array ({precision})"
                
            elif opcode == "syst_ld_w":
                # syst_ld_w rs1
                rs1 = operands[0]
                addr = self.registers[rs1]
                weights = self.memory.get(addr, None)
                if weights is None:
                    raise ValueError(f"No weights data found at memory address {addr}")
                if self.sim is None:
                    raise ValueError("Accelerator not configured. Call syst_cfg first.")
                    
                self.sim.array.load_weights(weights)
                trace_entry["accelerator_event"] = f"Loaded weights into systolic array from address {addr}"
                
            elif opcode == "syst_ld_a":
                # syst_ld_a rs1
                rs1 = operands[0]
                addr = self.registers[rs1]
                activations = self.memory.get(addr, None)
                if activations is None:
                    raise ValueError(f"No activation data found at memory address {addr}")
                if self.sim is None:
                    raise ValueError("Accelerator not configured. Call syst_cfg first.")
                    
                self.sim.array.load_activations(activations)
                trace_entry["accelerator_event"] = f"Loaded activations into systolic array from address {addr}"
                
            elif opcode == "syst_step":
                # syst_step rs1_act_addr, rs2_wt_addr
                rs1, rs2 = operands[0], operands[1]
                act_addr = self.registers[rs1]
                wt_addr = self.registers[rs2]
                
                act_in = self.memory.get(act_addr, [0.0] * self.sim.array_rows)
                wt_in = self.memory.get(wt_addr, [0.0] * self.sim.array_cols)
                
                self.sim.array.step(act_in, wt_in)
                self.sim.total_cycles += 1
                self.sim.history.append(self.sim.array.get_pe_activity())
                trace_entry["accelerator_event"] = f"Executed single cycle step"

            elif opcode == "syst_exec":
                # syst_exec rd, rs1, rs2, M, K, N
                rd, rs1, rs2 = operands[0], operands[1], operands[2]
                M, K, N = int(operands[3]), int(operands[4]), int(operands[5])
                
                addr_A = self.registers[rs1]
                addr_B = self.registers[rs2]
                addr_C = self.registers[rd]
                
                A = self.memory.get(addr_A, None)
                B = self.memory.get(addr_B, None)
                
                if A is None or B is None:
                    raise ValueError(f"Missing matrix data at address {addr_A} or {addr_B}")
                if self.sim is None:
                    raise ValueError("Accelerator not configured. Call syst_cfg first.")
                
                # Execute multiplication on hardware simulator
                C = self.sim.run(A, B)
                self.memory[addr_C] = C
                trace_entry["accelerator_event"] = f"Completed accelerator GEMM computation of shape ({M}x{K} * {K}x{N}) -> Saved C to address {addr_C}"
                
            else:
                raise ValueError(f"Unknown opcode: {opcode}")
                
        except Exception as e:
            trace_entry["status"] = f"Error: {str(e)}"
            self.execution_trace.append(trace_entry)
            raise e

        self.execution_trace.append(trace_entry)
        self.pc += 4

    def execute_program(self, asm_code):
        """
        Execute a complete multi-line assembly program string.
        """
        self.pc = 0
        self.execution_trace = []
        
        lines = asm_code.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self.execute_instruction(line)
            
        return self.execution_trace
