"""
Systolic Array Grid Module.
"""
from src.pe import ProcessingElement

class SystolicArray:
    def __init__(self, rows, cols, dataflow="OS"):
        self.rows = rows
        self.cols = cols
        self.dataflow = dataflow
        self.grid = [[ProcessingElement(r, c, dataflow=dataflow) for c in range(cols)] for r in range(rows)]
        
    def load_weights(self, weights_matrix):
        """
        Pre-load weights into the PEs. Used for Weight Stationary (WS) dataflow.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c].load_weight(weights_matrix[r][c])

    def load_activations(self, activations_matrix):
        """
        Pre-load activations into the PEs. Used for Row Stationary (RS) dataflow.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c].load_activation(activations_matrix[r][c])

    def step(self, activation_inputs, secondary_inputs):
        """
        Advance the simulation by one clock cycle.
        
        Args:
            activation_inputs: List of length `rows` fed to the left edge of the array.
            secondary_inputs: List of length `cols` fed to the top edge of the array.
                              (weights for OS, partial sums for WS, weights for RS)
        """
        if len(activation_inputs) != self.rows:
            raise ValueError(f"Expected {self.rows} activation inputs, got {len(activation_inputs)}")
        if len(secondary_inputs) != self.cols:
            raise ValueError(f"Expected {self.cols} secondary inputs, got {len(secondary_inputs)}")
            
        # Phase 1: Compute (Combinational logic)
        for r in range(self.rows):
            for c in range(self.cols):
                # Activation/Partial sum comes from the left
                act_in = activation_inputs[r] if c == 0 else self.grid[r][c-1].activation_out
                
                # Secondary input (weight or partial sum) comes from the top
                if self.dataflow in ["OS", "RS"]:
                    sec_in = secondary_inputs[c] if r == 0 else self.grid[r-1][c].weight_out
                elif self.dataflow == "WS":
                    sec_in = secondary_inputs[c] if r == 0 else self.grid[r-1][c].partial_sum_out
                else:
                    sec_in = 0.0
                
                self.grid[r][c].compute(act_in, sec_in)
                
        # Phase 2: Update (Clock edge latching)
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c].update()
                
    def get_partial_sums(self):
        """
        Return the C matrix currently held in the PEs.
        """
        return [[self.grid[r][c].partial_sum for c in range(self.cols)] for r in range(self.rows)]
        
    def get_pe_activity(self):
        """
        Return a boolean grid indicating which PEs are active (non-zero inputs).
        This is useful for utilization metrics and visualization.
        """
        activity = []
        for r in range(self.rows):
            row_activity = []
            for c in range(self.cols):
                if self.dataflow == "OS":
                    act_active = self.grid[r][c].activation_out != 0.0
                    sec_active = self.grid[r][c].weight_out != 0.0
                elif self.dataflow == "WS":
                    act_active = self.grid[r][c].activation_out != 0.0
                    sec_active = self.grid[r][c].weight != 0.0
                elif self.dataflow == "RS":
                    act_active = self.grid[r][c].activation != 0.0
                    sec_active = self.grid[r][c].weight_out != 0.0
                else:
                    act_active = False
                    sec_active = False
                row_activity.append(1 if act_active and sec_active else 0)
            activity.append(row_activity)
        return activity
        
    def reset(self):
        """
        Reset the entire array state.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c].reset()
