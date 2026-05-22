"""
Processing Element (PE) Module.
"""

class ProcessingElement:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        
        # Internal state
        self.partial_sum = 0.0
        
        # Current cycle outputs to neighbors
        self.activation_out = 0.0
        self.weight_out = 0.0
        
        # Next cycle state (double buffering for cycle-accurate simulation)
        self._next_activation_out = 0.0
        self._next_weight_out = 0.0
        self._next_partial_sum = 0.0
        
    def compute(self, activation_in, weight_in):
        """
        Compute MAC operation and register inputs for forwarding.
        """
        # Compute MAC
        self._next_partial_sum = self.partial_sum + (activation_in * weight_in)
        # Pass through inputs for next cycle
        self._next_activation_out = activation_in
        self._next_weight_out = weight_in
        
    def update(self):
        """
        Clock edge: latch computed values into current state.
        """
        self.partial_sum = self._next_partial_sum
        self.activation_out = self._next_activation_out
        self.weight_out = self._next_weight_out
        
    def reset(self):
        """
        Reset PE state for a new computation.
        """
        self.partial_sum = 0.0
        self.activation_out = 0.0
        self.weight_out = 0.0
        self._next_activation_out = 0.0
        self._next_weight_out = 0.0
        self._next_partial_sum = 0.0
