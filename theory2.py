"""Theoretical calculations of amplifier noise signals.

TODO: Add preamp capacitance.

Notes:
    Although the feedback resistance can be changed for the voltage
    noise response, the current noise response assumes a preamp
    transimpedance of rho.

Calculates electronic noise signal(s) for one center FCalPulse tube
segment. *Without* shaping from digitizer front-end amp. All units are
SI unless otherwise stated.

Written by Anson Kost, adapted from code by Prof. John Rutherfoord.
July 2019.
"""

import math
import warnings

import scipy.integrate
from scipy.constants import pi, Boltzmann, zero_Celsius

R = 26  # Series noise source impedance.
Q = 770  # Parallel noise source impedance.

C = 6e-12  # Estimate of tube cals capacitance.
C_s = 5.7e-12  # Shower cals capacitance.

A = 39  # Opamp amplification.
A2 = 2.2  # "Second stage" FAMP amplification.

T = 4e-9  # Cable time delay.
rho = 50  # Cable characteristic impedance.
rho_amp = rho  # FAMP transimpedance.

omega_digitizer = 2 * pi * 250e6  # Digitizer high frequency cutoff.
tau_r = 1 / omega_digitizer  # Digitizer time constant.
tau_a = 0  # A time constant.
R_c1, R_c2 = 1e-6, 1e-6  # Low frequency cutoff resistances.

# Low frequency cutoff time constants.
tau_c1, tau_c2 = rho * R_c1, rho * R_c2

omega_lims = (0, 1 * omega_digitizer)  # omega integration limits.

# Overall factors that don't change over the integration.
_constant_factor = (2 / pi) * Boltzmann * zero_Celsius


def noise(Z_end_fn=None, printout=True):
    if not Z_end_fn:
        def Z_end_fn(omega):
            return 0

    integration_result, error = scipy.integrate.quad(_integrand,
                                                     omega_lims[0],
                                                     omega_lims[1],
                                                     args=(
                                                         Z_end_fn, R, Q, T,
                                                         rho_amp, tau_a, tau_r,
                                                         tau_c1, tau_c2))

    # A naive check on the integration error.
    if abs(error / integration_result) > 1e-7: warnings.warn(
        "The integration error is relatively big!")

    # Preamp input noise RMS voltage.
    input_V = math.sqrt(_constant_factor * integration_result)

    # Ditto, at the output.
    output_V = A2 * A * input_V

    # Effective preamp input noise RMS current.
    input_I_eff = input_V / rho_amp

    if printout:
        print(
            "Noise signals:\n"
            f"Input RMS voltage: {round(1000 * input_V, 4)} mV.\n"
            "Effective input RMS current (\"ENI\"):"
            f" {round(1e9 * input_I_eff, 4)} nA.\n"
            f"Output RMS voltage: {round(1000 * output_V, 4)} mV.\n"
        )

    return output_V


def _integrand(omega, Z_end_fn, R, Q, T, rho_amp, tau_a, tau_r, tau_c1,
               tau_c2):
    Z_cabl = Z_cable(omega * T, Z_end_fn(omega), rho)
    return (
                   R * abs(F(Z_cabl, A, rho_amp)) ** 2
                   + (rho_amp ** 2 / Q) * abs(G(Z_cabl, rho_amp)) ** 2
           ) * H_squared(omega, tau_a, tau_r, tau_c1, tau_c2)


def F(Z_cabl, A, rho_amp):
    return (Z_cabl + (A + 1) * rho_amp) / (
            (A - 1) * Z_cabl - (A + 1) * rho_amp)


def G(Z_cabl, rho_amp):
    return Z_cabl / (Z_cabl + rho_amp)


def Z_cable(theta, Z_end, rho):
    """ "Input" impedance of an ideal lossless transmission line in
    terms of the impedance Z_end at the other end."""
    tan_theta = math.tan(theta)
    return rho * (
            Z_end + 1.j * rho * tan_theta
    ) / (rho + 1.j * Z_end * tan_theta)


def H_squared(omega, tau_a, tau_r, tau_c1, tau_c2):
    omega_tau_c1_squared = (omega * tau_c1) ** 2
    omega_tau_c2_squared = (omega * tau_c2) ** 2
    return 1 / (
            (1 + (omega * tau_a) ** 2)
            * (1 + (omega * tau_r) ** 2)
            * omega_tau_c1_squared / (1 + omega_tau_c1_squared)
            * omega_tau_c2_squared / (1 + omega_tau_c2_squared)
    )


def Z_capacitance(omega, C):
    return - 1.j / (omega * C)


def Z_inductance(omega, L):
    return 1.j * omega * L


import numpy as np
from matplotlib import pyplot as plt
import seaborn

seaborn.set()

def plot():
    Cs = np.linspace(1e-14, 100e-12, 1000)
    noises = np.array([
        noise(lambda omega: Z_capacitance(omega, C), printout=False)
        for C in Cs
    ])
    plt.plot(1e12 * Cs, 1000 * noises)
    plt.title('Noise vs. Load Capacitance')
    plt.xlabel('Capacitance (pF)')
    plt.ylabel('Noise (mV)')
    plt.show()

noise(lambda omega: Z_capacitance(omega, C))
noise(lambda omega: Z_capacitance(omega, C_s))
plot()
