"""Theoretical calculations of amplifier noise signals.

Calculates electronic noise signal(s) for one center FCalPulse tube
segment. *Without* shaping from digitzer front-end amp. All units are SI
unless otherwise stated.

omega is angular frequency.

TODO: Add preamp capacitance.

Written by Anson Kost, adapted from code by Prof. John Rutherfoord. July 2019.
"""

# From the Python Standard Library (if you're curious).
import math
import cmath
import warnings

# From Anaconda.
import numpy
from scipy import integrate
from scipy.constants import pi, Boltzmann, zero_Celsius

A = 39      # Opamp amplification.
A2 = 2.2    # "Second stage" preamp amplification.

T = 4e-9    # Cable time delay.
rho = 50    # Cable characteristic impedance.
C = 6e-12   # Detector capacitance.

R = 26      # Series noise source impedance.
Q = 770     # Parallel noise source impedance.

tau_a = 0   # ?

omega_digitzer = 2 * pi * 250e6     # Digitizer high frequency cutoff.

# For the frequency filter function (H_squared).
tau_r = 1 / omega_digitzer

tau = rho * C                       # Detector time constant.

omega_lims = (0, omega_digitzer)    # omega integration limits.


def F_squared(omega):
    """Square magnitude of the series voltage "transfer function"
    (noise to amp output).
    """
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
    """The parallel current "transfer function." """
    theta_i = omega * tau * 1j
    phi_i = omega * T * 1j
    return 1 / 2 * (
            1 + ((1 - theta_i) / (1 + theta_i)) * cmath.exp(-2 * phi_i)
    )


def G_squared(omega):
    """Square magnitude of the parallel current "transfer function." """
    return abs(G(omega)) ** 2


def H_squared(omega):
    """Square magnitude of the frequency filter function."""
    return 1 / (
            (1 + (omega * tau_a) ** 2) * (1 + (omega * tau_r) ** 2)
    ) * H_squared_heaviside(omega)


def H_squared_heaviside(omega):
    return numpy.heaviside(omega_digitzer - omega, 1 / 2)


def integrand(omega):
    """Just the factors that change over the integration.

    H_squared is included here since it decides the range of frequencies
    we care about.
    """
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

# Overall factors that don't change over the integration.
constant_factor = (2 / pi) * Boltzmann * zero_Celsius

# Preamp input noise RMS voltage.
input_V = math.sqrt(constant_factor * integration_result)

# Ditto, at the output.
output_V = A2 * A * input_V

# Effective preamp input noise RMS current.
input_I_eff = input_V / rho

print(
    "Noise signals:\n"
    f"Input RMS voltage: {round(1000 * input_V, 4)} mV.\n"
    "Effective input RMS current (\"ENI\"):"
    f" {round(1e9 * input_I_eff, 4)} nA.\n"
    f"Output RMS voltage: {round(1000 * output_V, 4)} mV.\n"
)
