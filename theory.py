"""Theoretical calculations of amplifier noise signals.

TODO: Add preamp capacitance.
TODO: Update frequency function.

Calculates electronic noise signal(s) for one center FCalPulse tube
segment. *Without* shaping from digitizer front-end amp. All units are
SI unless otherwise stated.

`omega` is angular frequency.

Note about abstracting the physics formulas:
This script implements the F_squared etc. functions directly, but, since
Python can handle complex numbers, a much more general approach is
possible. For example, the F_squared function could take an arbitrary
transmission line input impedance as a parameter, and that input
impedance could be calculated later in some other function. Abstractions
like these can replace some of the algebra and can be taken as far as
one wants. theory2.py takes a more abstract approach.

Written by Anson Kost, adapted from code by Prof. John Rutherfoord.
July 2019.
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

omega_digitizer = 2 * pi * 250e6     # Digitizer high frequency cutoff.

# For the frequency filter function (H_squared).
tau_r = 1 / omega_digitizer

tau = rho * C                        # Detector time constant.

omega_lims = (0, omega_digitizer)    # omega integration limits.


def F_squared(omega):
    """Square magnitude of the series voltage "transfer function"
    (noise to amp output).

    For a detector capacitance at the end of the transmission line.
    """
    phi = omega * tau

    # alpha is an angle that depends on the detector time constant
    # (rho * C), but not the cable length.
    alpha = math.acos((1 - phi ** 2) / (1 + phi ** 2))

    return (
            1 / (2 * (A + 1)) ** 2
            * (
                    A ** 2
                    + (A + 2) ** 2
                    - A * (A + 2) * math.cos(2 * omega * T + alpha)
            )
    )


def F_squared_short(omega):
    """For a short at the end of the transmission line."""
    theta = omega * T
    return ((A + 1) ** 2 + math.tan(theta) ** 2) / (
            ((A - 1) * math.tan(theta)) ** 2 + (A + 1) ** 2
    )


def F_squared_open(omega):
    """For an open circuit at the end of the transmission line.

    (The impedance of a transmission line with an open circuit at the
    end is the same as that of a line with a short at the end and which
    is an additional 1/4 wavelength longer.)
    """
    theta = omega * T
    return ((A + 1) ** 2 + 1 / math.tan(theta) ** 2) / (
            ((A - 1) / math.tan(theta)) ** 2 + (A + 1) ** 2
    )


def G(omega):
    """The parallel current "transfer function." """
    theta_imag = omega * T * 1j
    phi_imag = omega * tau * 1j
    return 1 / 2 * (
            1 + ((1 - phi_imag) / (1 + phi_imag)) * cmath.exp(-2 * theta_imag)
    )


def G_squared(omega):
    """Square magnitude of the parallel current "transfer function." """
    return abs(G(omega)) ** 2


def G_squared_short(omega):
    return math.sin(omega * T) ** 2


def G_squared_open(omega):
    """pi/4 phase shift relative to a shorted line."""
    return math.cos(omega * T) ** 2


def H_squared(omega):
    """Square magnitude of the frequency filter function."""
    return 1 / (
            (1 + (omega * tau_a) ** 2) * (1 + (omega * tau_r) ** 2)
    ) * H_squared_heaviside(omega)


def H_squared_heaviside(omega):
    return numpy.heaviside(omega_digitizer - omega, 1 / 2)


def integrand(omega):
    """Just the factors that change over the integration.

    H_squared is included here since it decides the range of frequencies
    we care about.
    """
    return (
                   R * F_squared_short(omega)
                   + (rho ** 2 / Q) * G_squared_short(omega)
           ) * H_squared(omega)


# Perform the integration using SciPy's integrator (which uses the
# Fortran QUADPACK package).
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
