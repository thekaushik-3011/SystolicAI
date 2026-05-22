"""
Synthesizable Verilog RTL Generator.
"""

class RTLGenerator:
    def __init__(self, rows, cols, dataflow="OS", precision="FP32"):
        self.rows = rows
        self.cols = cols
        self.dataflow = dataflow
        self.precision = precision
        
        # Determine bitwidths
        if precision == "INT8":
            self.input_width = 8
            self.accum_width = 32
        else:
            self.input_width = 32
            self.accum_width = 32

    def generate_pe(self):
        """
        Generate Verilog code for a single Processing Element (PE).
        """
        if self.dataflow == "OS":
            return f"""// Single Processing Element for Output Stationary (OS) Dataflow
module processing_element #(
    parameter INPUT_WIDTH = {self.input_width},
    parameter ACCUM_WIDTH = {self.accum_width}
) (
    input clk,
    input rst,
    input enable,
    input signed [INPUT_WIDTH-1:0] activation_in,
    input signed [INPUT_WIDTH-1:0] weight_in,
    output reg signed [INPUT_WIDTH-1:0] activation_out,
    output reg signed [INPUT_WIDTH-1:0] weight_out,
    output reg signed [ACCUM_WIDTH-1:0] psum_out
);
    reg signed [ACCUM_WIDTH-1:0] accumulator;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            accumulator <= 0;
            activation_out <= 0;
            weight_out <= 0;
        end else if (enable) begin
            accumulator <= accumulator + (activation_in * weight_in);
            activation_out <= activation_in;
            weight_out <= weight_in;
        end
    end

    always @(*) begin
        psum_out = accumulator;
    end
endmodule
"""
        elif self.dataflow == "WS":
            return f"""// Single Processing Element for Weight Stationary (WS) Dataflow
module processing_element #(
    parameter INPUT_WIDTH = {self.input_width},
    parameter ACCUM_WIDTH = {self.accum_width}
) (
    input clk,
    input rst,
    input enable,
    input weight_load,
    input signed [INPUT_WIDTH-1:0] activation_in,
    input signed [INPUT_WIDTH-1:0] weight_in,
    input signed [ACCUM_WIDTH-1:0] psum_in,
    output reg signed [INPUT_WIDTH-1:0] activation_out,
    output reg signed [INPUT_WIDTH-1:0] weight_out,
    output reg signed [ACCUM_WIDTH-1:0] psum_out
);
    reg signed [INPUT_WIDTH-1:0] weight_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            weight_reg <= 0;
            activation_out <= 0;
            weight_out <= 0;
            psum_out <= 0;
        end else if (enable) begin
            if (weight_load) begin
                weight_reg <= weight_in;
                weight_out <= weight_in; // Propagate down for loading during config
            end else begin
                activation_out <= activation_in;
                weight_out <= 0;
                psum_out <= psum_in + (activation_in * weight_reg);
            end
        end
    end
endmodule
"""
        else: # RS
            return f"""// Single Processing Element for Row Stationary (RS) Dataflow
module processing_element #(
    parameter INPUT_WIDTH = {self.input_width},
    parameter ACCUM_WIDTH = {self.accum_width}
) (
    input clk,
    input rst,
    input enable,
    input act_load,
    input signed [INPUT_WIDTH-1:0] activation_in,
    input signed [INPUT_WIDTH-1:0] weight_in,
    input signed [ACCUM_WIDTH-1:0] psum_in,
    output reg signed [INPUT_WIDTH-1:0] activation_out,
    output reg signed [INPUT_WIDTH-1:0] weight_out,
    output reg signed [ACCUM_WIDTH-1:0] psum_out
);
    reg signed [INPUT_WIDTH-1:0] act_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            act_reg <= 0;
            activation_out <= 0;
            weight_out <= 0;
            psum_out <= 0;
        end else if (enable) begin
            if (act_load) begin
                act_reg <= activation_in;
                activation_out <= activation_in; // Propagate right for loading
            end else begin
                weight_out <= weight_in;
                activation_out <= psum_in + (act_reg * weight_in); // Stream PSum out horizontally
                psum_out <= psum_in + (act_reg * weight_in);
            end
        end
    end
endmodule
"""

    def generate_array(self):
        """
        Generate Verilog code for the interconnected Systolic Array Grid.
        """
        pe_instantiations = []
        wire_declarations = []
        
        # Generate wire declarations for mesh interconnections
        for r in range(self.rows):
            for c in range(self.cols):
                # Horizontal wires (activations or Horizontal PSums)
                wire_declarations.append(f"    wire signed [{self.input_width}-1:0] act_w_{r}_{c}_{c+1};")
                # Vertical wires (weights or Vertical PSums)
                wire_declarations.append(f"    wire signed [{self.input_width}-1:0] wt_w_{r}_{r+1}_{c};")
                # PSum wires
                wire_declarations.append(f"    wire signed [{self.accum_width}-1:0] psum_w_{r}_{r+1}_{c};")
                if self.dataflow == "OS":
                    wire_declarations.append(f"    wire signed [{self.accum_width}-1:0] pe_accum_w_{r}_{c};")
        
        wire_str = "\n".join(wire_declarations)
        
        # Instantiate PEs
        for r in range(self.rows):
            for c in range(self.cols):
                # Inputs for OS, WS, RS
                act_in = f"act_in_{r}" if c == 0 else f"act_w_{r}_{c-1}_{c}"
                wt_in = f"wt_in_{c}" if r == 0 else f"wt_w_{r-1}_{r}_{c}"
                
                # Outputs
                act_out = f"act_w_{r}_{c}_{c+1}"
                wt_out = f"wt_w_{r}_{r}_{c+1}" if self.dataflow == "WS" else f"wt_w_{r}_{r+1}_{c}"
                
                if self.dataflow == "OS":
                    inst = f"""    processing_element pe_{r}_{c} (
        .clk(clk),
        .rst(rst),
        .enable(enable),
        .activation_in({act_in}),
        .weight_in({wt_in}),
        .activation_out({act_out}),
        .weight_out(wt_w_{r}_{r+1}_{c}),
        .psum_out(pe_accum_w_{r}_{c})
    );"""
                elif self.dataflow == "WS":
                    psum_in = f"32'sd0" if r == 0 else f"psum_w_{r-1}_{r}_{c}"
                    inst = f"""    processing_element pe_{r}_{c} (
        .clk(clk),
        .rst(rst),
        .enable(enable),
        .weight_load(weight_load),
        .activation_in({act_in}),
        .weight_in({wt_in}),
        .psum_in({psum_in}),
        .activation_out({act_out}),
        .weight_out(wt_w_{r}_{r+1}_{c}),
        .psum_out(psum_w_{r}_{r+1}_{c})
    );"""
                else: # RS
                    # Row Stationary streams PSum horizontally from left to right, weights down
                    psum_in = f"32'sd0" if c == 0 else f"psum_w_{r}_{c-1}_{c}"
                    inst = f"""    processing_element pe_{r}_{c} (
        .clk(clk),
        .rst(rst),
        .enable(enable),
        .act_load(act_load),
        .activation_in({act_in}),
        .weight_in({wt_in}),
        .psum_in({psum_in}),
        .activation_out({act_out}),
        .weight_out(wt_w_{r}_{r+1}_{c}),
        .psum_out(psum_w_{r}_{r}_{r+1}) // Wire mapping to horizontal neighbour
    );"""
                pe_instantiations.append(inst)

        pe_str = "\n\n".join(pe_instantiations)
        
        # Build Array Inputs / Outputs Port list
        ports = ["input clk", "input rst", "input enable"]
        if self.dataflow == "WS":
            ports.append("input weight_load")
        elif self.dataflow == "RS":
            ports.append("input act_load")
            
        for r in range(self.rows):
            ports.append(f"    input signed [{self.input_width}-1:0] act_in_{r}")
        for c in range(self.cols):
            ports.append(f"    input signed [{self.input_width}-1:0] wt_in_{c}")
            
        if self.dataflow == "OS":
            for r in range(self.rows):
                for c in range(self.cols):
                    ports.append(f"    output signed [{self.accum_width}-1:0] psum_out_{r}_{c}")
        elif self.dataflow == "WS":
            for c in range(self.cols):
                ports.append(f"    output signed [{self.accum_width}-1:0] psum_out_{c}")
        else: # RS
            for r in range(self.rows):
                ports.append(f"    output signed [{self.accum_width}-1:0] psum_out_{r}")

        ports_str = ",\n".join(ports)
        
        # Assign outputs
        assignments = []
        if self.dataflow == "OS":
            for r in range(self.rows):
                for c in range(self.cols):
                    assignments.append(f"    assign psum_out_{r}_{c} = pe_accum_w_{r}_{c};")
        elif self.dataflow == "WS":
            for c in range(self.cols):
                assignments.append(f"    assign psum_out_{c} = psum_w_{self.rows-1}_{self.rows}_{c};")
        else: # RS
            for r in range(self.rows):
                assignments.append(f"    assign psum_out_{r} = psum_w_{r}_{self.cols-1}_{self.cols};")
                
        assign_str = "\n".join(assignments)

        return f"""// {self.rows}x{self.cols} Systolic Array Grid with {self.dataflow} Dataflow
module systolic_array (
{ports_str}
);

{wire_str}

{pe_str}

{assign_str}

endmodule
"""

    def generate_complete_system(self):
        """
        Combine PE and Array code into a single Verilog package string.
        """
        pe_code = self.generate_pe()
        array_code = self.generate_array()
        
        return f"""// ====================================================================
// SystolicAI Auto-Generated Synthesizable Verilog Hardware Backend
// Config: {self.rows}x{self.cols} grid | {self.dataflow} dataflow | {self.precision} precision
// ====================================================================

{pe_code}

// --------------------------------------------------------------------

{array_code}
"""
