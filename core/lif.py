"""
This file defines the LIF class along with associated functionality.
"""

from typing import List, NamedTuple

import numpy as np


class LIFParams(NamedTuple):
    dt: float = 0.05 # msec
    c_m: float = 58.72 # pF
    g_l: float = 9.43 # 1/GOhm (nS)
    v_l: float = -78.85 # mV
    v_th: float = -51.68 # mV
    k_j: np.ndarray = np.array([0.003, 0.1]) # msec^-1
    f_j: np.ndarray = np.array([-9.18, -198.94]) # pA
    b_j: np.ndarray = np.array([1.0, 1.0]) # no unit

class LIF:
    """Initializes LIF neuron with parameters provided."""
    def __init__(self, params: LIFParams):
        self.dt = params.dt
        self.c_m = params.c_m
        self.g_l = params.g_l
        self.v_l = params.v_l
        self.v_th = params.v_th
        self.k_j = params.k_j
        self.b_j = params.b_j
        self.f_j = params.f_j

        # States to track
        self.v = self.v_l
        self.i_j = np.zeros(2)

    def forward(self, i_ext):
        """
        Computes response to external/applied current using following equations.
        
        If V > V_th:
        V(t) = V_l # Reset voltage
        I_j(t) = I_j(t-1) * f_j + b_j
        
        Else:
        V(t) = V(t) + (dt / C_m) * (I_ext(t) + sum(I_j(t-1)) - g_l * (V(t-1) - V_l)
        I_j(t) = I_j(t-1) * (1 - k_j * dt)
        
        Arguments:
        - i_ext: list
          list of externally applied currents (pA)
        - dt: float
          timestep duration to use in forward euler
        Returns:
        - s: list
          list of 1s and 0s pertaining to whether a neuron spiked
          in each timestep corresponding to the applied current
        """
        s = []
        for i_step in i_ext:
            if self.v > self.v_th:  # Spiking case
                self.v = self.v_l
                self.i_j = self.i_j * self.f_j + self.b_j
                s.append(1)
            else:  # Base case
                dv = (1 / self.c_m) * (i_step + np.sum(self.i_j) - self.g_l * (self.v - self.v_l))
                self.v += dv * self.dt
                self.i_j = self.i_j - self.k_j * self.i_j * self.dt
                s.append(0)
        return s