"""
Processing Element (PE) Module.
"""

class ProcessingElement:
    def __init__(self, row, col, dataflow="OS"):
        self.row = row
        self.col = col
        self.dataflow = dataflow
        
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
            weight_in = secondary_in
            # Compute MAC and accumulate locally
            self._next_partial_sum = self.partial_sum + (activation_in * weight_in)
            # Pass through inputs for next cycle
            self._next_activation_out = activation_in
            self._next_weight_out = weight_in
            
        elif self.dataflow == "WS":
            partial_sum_in = secondary_in
            # Compute MAC and pass partial sum down
            self._next_partial_sum_out = partial_sum_in + (activation_in * self.weight)
            # Pass activation right
            self._next_activation_out = activation_in

        elif self.dataflow == "RS":
            weight_in = secondary_in
            partial_sum_in = activation_in
            # Compute MAC and pass partial sum right
            self._next_partial_sum_out = partial_sum_in + (self.activation * weight_in)
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
