"""
Theoretical calculations of amplifier noise signals.

Calculates electronic noise signal(s) for one center FCalPulse tube
segment. *Without* shaping from digitzer front-end amp. All units are SI
unless otherwise stated.

omega is angular frequency.

TODO: Add preamp capacitance and additional amp gain.

Written by Anson Kost, adapted from code by Prof. John Rutherfoord.
"""

import math
import cmath
import warnings

import numpy
from scipy import integrate
from scipy.constants import pi, Boltzmann, zero_Celsius

A = 39                  # Opamp amplification.
T = 4e-9                # Cable time delay.
rho = 50                # Cable characteristic impedance.
C = 6e-12               # Detector capacitance.

R = 26                  # Series noise source impedance.
Q = 770                 # Parallel noise source impedance.

omega_lims = (0, 1e10)  # omega integration limits.

tau = rho * C           # Detector time constant.


def F_squared(omega):
    """Square magnitude of the series "transfer function"
    (noise to amp output)."""
    theta = omega * tau

    # alpha is an angle that depends on the detector time constant
    # (rho * C), but not the cable length.
    alpha = math.acos((1 - theta ** 2) / (1 + theta ** 2))

    return (
            1 / (2 * (A + 1)) ** 2
            * (
                    A ** 2
                    + (A + 2) ** 2
                    - A * (A + 2) * math.cos(2 * omega * T + alpha)
            )
    )


def G(omega):
    """The parallel "transfer function." """
    theta_i = omega * tau * 1j
    phi_i = omega * T * 1j
    return 1 / 2 * (
            1 + ((1 - theta_i) / (1 + theta_i)) * cmath.exp(-2 * phi_i)
    )


def G_squared(omega):
    """Square magnitude of the parallel "transfer function." """
    return abs(G(omega)) ** 2


def H_squared(omega):
    """Square magnitude of the frequency filter function."""
    return numpy.heaviside(omega_lims[1] - omega, 1 / 2)


def integrand(omega):
    """Just the factors that change over the integration."""
    return (
                   R * F_squared(omega)
                   + (rho ** 2 / Q) * G_squared(omega)
           ) * H_squared(omega)


integration_result, error = integrate.quad(
    integrand, omega_lims[0], omega_lims[1]
)

# A naive check on the integration error.
if abs(error / integration_result) > 1e-7:
    warnings.warn("The integration error is relatively big!")

constant_factor = (2 / pi) * Boltzmann * zero_Celsius * A ** 2
noise_rms_voltage = math.sqrt(constant_factor * integration_result)

print(f"Noise signal RMS voltage: {round(1000 * noise_rms_voltage, 4)} mV.")
