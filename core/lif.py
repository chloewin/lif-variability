class LIF:
    def __init__(self, c_m, g_l, v_l, v_th):
        self.c_m = c_m
        self.g_l = g_l
        self.v_l = v_l
        self.v_th = v_th
        self.v = self.v_l
    
    def forward(self, i_ext, dt):
        # Computes response to external/applied current.
        #
        # Arguments:
        # - i_ext: list
        #   list of externally applied currents (pA)
        # - dt: float
        #   timestep duration to use in forward euler
        # Returns:
        # - s: list
        #   list of 1s and 0s pertaining to whether a neuron spiked
        #   in each timestep corresponding to the applied current
        s = []
        for i_step in i_ext:
            if self.v > self.thresh:
                self.v = self.v_l
                s.append(1)
            else:
                dv = (1 / self.c_m) * (i_step - self.g_l * (self.v_m - self.v_l))
                self.v += dv
                s.append(0)
        return s