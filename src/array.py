"""
Systolic Array Grid Module.
"""
from src.pe import ProcessingElement

class SystolicArray:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[ProcessingElement(r, c) for c in range(cols)] for r in range(rows)]
        
    def step(self, activation_inputs, weight_inputs):
        """
        Advance the simulation by one clock cycle.
        
        Args:
            activation_inputs: List of length `rows` fed to the left edge of the array.
            weight_inputs: List of length `cols` fed to the top edge of the array.
        """
        if len(activation_inputs) != self.rows:
            raise ValueError(f"Expected {self.rows} activation inputs, got {len(activation_inputs)}")
        if len(weight_inputs) != self.cols:
            raise ValueError(f"Expected {self.cols} weight inputs, got {len(weight_inputs)}")
            
        # Phase 1: Compute (Combinational logic)
        for r in range(self.rows):
            for c in range(self.cols):
                # Activation comes from the left
                act_in = activation_inputs[r] if c == 0 else self.grid[r][c-1].activation_out
                # Weight comes from the top
                wt_in = weight_inputs[c] if r == 0 else self.grid[r-1][c].weight_out
                
                self.grid[r][c].compute(act_in, wt_in)
                
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
                # A PE is active if it passes through a non-zero activation and weight this cycle.
                # Since 'update' was called, we check the latched outputs.
                act_active = self.grid[r][c].activation_out != 0.0
                wt_active = self.grid[r][c].weight_out != 0.0
                row_activity.append(1 if act_active and wt_active else 0)
            activity.append(row_activity)
        return activity
        
    def reset(self):
        """
        Reset the entire array state.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c].reset()
