"""
Processing Element (PE) Module.
"""
import numpy as np

class ProcessingElement:
    def __init__(self, row, col, dataflow="OS", precision="FP32"):
        self.row = row
        self.col = col
        self.dataflow = dataflow
        self.precision = precision
        
        # Internal state
        self.partial_sum = 0.0 # Used in OS
        self.weight = 0.0      # Used in WS
        self.activation = 0.0  # Used in RS
        
        # Current cycle outputs to neighbors
        self.activation_out = 0.0
        self.weight_out = 0.0
        self.partial_sum_out = 0.0
        
        # Next cycle state (double buffering for cycle-accurate simulation)
        self._next_activation_out = 0.0
        self._next_weight_out = 0.0
        self._next_partial_sum_out = 0.0
        self._next_partial_sum = 0.0
        
    def compute(self, activation_in, secondary_in):
        """
        Compute MAC operation and register inputs for forwarding.
        In OS: secondary_in is weight_in
        In WS: secondary_in is partial_sum_in
        In RS: secondary_in is weight_in, activation_in is partial_sum_in
        """
        if self.dataflow == "OS":
            if self.precision == "INT8":
                act_in = float(np.clip(np.round(activation_in), -128, 127))
                weight_in = float(np.clip(np.round(secondary_in), -128, 127))
                sum_val = float(np.clip(np.round(self.partial_sum + (act_in * weight_in)), -2147483648, 2147483647))
            else:
                act_in = activation_in
                weight_in = secondary_in
                sum_val = self.partial_sum + (act_in * weight_in)
                
            self._next_partial_sum = sum_val
            # Pass through inputs for next cycle
            self._next_activation_out = act_in
            self._next_weight_out = weight_in
            
        elif self.dataflow == "WS":
            if self.precision == "INT8":
                act_in = float(np.clip(np.round(activation_in), -128, 127))
                self.weight = float(np.clip(np.round(self.weight), -128, 127))
                partial_sum_in = float(np.clip(np.round(secondary_in), -2147483648, 2147483647))
                sum_val = float(np.clip(np.round(partial_sum_in + (act_in * self.weight)), -2147483648, 2147483647))
            else:
                act_in = activation_in
                partial_sum_in = secondary_in
                sum_val = partial_sum_in + (act_in * self.weight)
                
            self._next_partial_sum_out = sum_val
            # Pass activation right
            self._next_activation_out = act_in

        elif self.dataflow == "RS":
            if self.precision == "INT8":
                weight_in = float(np.clip(np.round(secondary_in), -128, 127))
                self.activation = float(np.clip(np.round(self.activation), -128, 127))
                partial_sum_in = float(np.clip(np.round(activation_in), -2147483648, 2147483647))
                sum_val = float(np.clip(np.round(partial_sum_in + (self.activation * weight_in)), -2147483648, 2147483647))
            else:
                weight_in = secondary_in
                partial_sum_in = activation_in
                sum_val = partial_sum_in + (self.activation * weight_in)
                
            self._next_partial_sum_out = sum_val
            # Pass weight down
            self._next_weight_out = weight_in
            # Pass partial sum right (using the horizontal activation_out wire)
            self._next_activation_out = self._next_partial_sum_out

    def load_weight(self, weight_val):
        """ Used in WS mode to pre-load weights """
        self.weight = weight_val

    def load_activation(self, activation_val):
        """ Used in RS mode to pre-load activations """
        self.activation = activation_val

    def update(self):
        """
        Clock edge: latch computed values into current state.
        """
        self.partial_sum = self._next_partial_sum
        self.activation_out = self._next_activation_out
        self.weight_out = self._next_weight_out
        self.partial_sum_out = self._next_partial_sum_out
        
    def reset(self):
        """
        Reset PE state for a new computation.
        """
        self.partial_sum = 0.0
        self.weight = 0.0
        self.activation = 0.0
        self.activation_out = 0.0
        self.weight_out = 0.0
        self.partial_sum_out = 0.0
        self._next_activation_out = 0.0
        self._next_weight_out = 0.0
        self._next_partial_sum_out = 0.0
        self._next_partial_sum = 0.0
