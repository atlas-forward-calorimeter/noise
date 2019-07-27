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


def noise(Z_end_fn=None, contributions='SP', printout=True):
    """The total integrated noise."""
    _check_contributions_string(contributions)

    if not Z_end_fn:
        def Z_end_fn(omega):
            return 0

    integration_result, error = 0, 0

    if 'S' in contributions:
        result, err = scipy.integrate.quad(
            _series_integrand, omega_lims[0], omega_lims[1], args=(
                Z_end_fn, A, T, rho, rho_amp, tau_a, tau_r, tau_c1, tau_c2
            )
        )
        _check_integration(result, err)

        integration_result += _series_constant_factor * result

    if 'P' in contributions:
        result, err = scipy.integrate.quad(
            _parallel_integrand, omega_lims[0], omega_lims[1], args=(
                Z_end_fn, T, rho, rho_amp, tau_a, tau_r, tau_c1, tau_c2
            )
        )
        _check_integration(result, err)

        integration_result += _parallel_constant_factor * result

    # Preamp input noise RMS voltage.
    input_V = _sqrt_constant_factor * math.sqrt(integration_result)

    output_V, input_I_eff = _input_V_calcs(input_V, A, A2, rho_amp)

    if printout:
        _printout(input_V, input_I_eff, output_V)

    return output_V


def noise_density(omega, Z_end_fn=None, contributions='SP'):
    """Square voltage of the noise per angular frequency at the preamp
    input.
    """
    _check_contributions_string(contributions)

    if not Z_end_fn:
        def Z_end_fn(omega):
            return 0

    density = 0

    if 'S' in contributions:
        density += _series_constant_factor * _series_integrand(
            omega, Z_end_fn, A, T, rho, rho_amp, tau_a, tau_r, tau_c1, tau_c2
        )

    if 'P' in contributions:
        density += _parallel_constant_factor * _parallel_integrand(
            omega, Z_end_fn, T, rho, rho_amp, tau_a, tau_r, tau_c1, tau_c2
        )

    return _constant_factor * density


def _input_V_calcs(input_V, A, A2, rho_amp):
    """Get the effective preamp input noise current (ENI) and the output
    noise voltage from the noise voltage at the preamp input."""
    output_V = A2 * A * input_V
    input_I_eff = input_V / rho_amp

    return output_V, input_I_eff


# Overall factors that are constant over the integration.
_constant_factor = (2 / pi) * Boltzmann * zero_Celsius
_sqrt_constant_factor = math.sqrt(_constant_factor)


def _integrand(omega, Z_end_fn, A, T, rho, rho_amp, tau_a, tau_r, tau_c1,
               tau_c2):
    """The sum of the series and parallel integrands with their
    individual constant factors. An overall constant factor is still
    excluded.
    """
    # return _series_constant_factor * _series_integrand(
    #     omega, Z_end_fn, A, T, rho, rho_amp, tau_a, tau_r, tau_c1, tau_c2
    # ) + _parallel_constant_factor * _parallel_integrand(
    #     omega, Z_end_fn, T, rho, rho_amp, tau_a, tau_r, tau_c1, tau_c2
    # )

    Z_cabl = Z_cable(omega * T, Z_end_fn(omega), rho)
    return (
                   R * abs(F(Z_cabl, A, rho_amp)) ** 2
                   + (rho_amp ** 2 / Q) * abs(G(Z_cabl, rho_amp)) ** 2
           ) * H_squared(omega, tau_a, tau_r, tau_c1, tau_c2)


_series_constant_factor = R


def _series_integrand(omega, Z_end_fn, A, T, rho, rho_amp, tau_a, tau_r,
                      tau_c1, tau_c2):
    # |F|^2 |H|^2
    return abs(F(
        Z_cable(omega * T, Z_end_fn(omega), rho),
        A,
        rho_amp
    )) ** 2 * H_squared(omega, tau_a, tau_r, tau_c1, tau_c2)


_parallel_constant_factor = rho_amp ** 2 / Q


def _parallel_integrand(omega, Z_end_fn, T, rho, rho_amp, tau_a, tau_r, tau_c1,
                        tau_c2):
    # |G|^2 |H|^2
    return abs(G(
        Z_cable(omega * T, Z_end_fn(omega), rho),
        rho_amp
    )) ** 2 * H_squared(omega, tau_a, tau_r, tau_c1, tau_c2)


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


def _check_contributions_string(string):
    """Check a string that represents which noise contributions to
    include in a calculation.
    """
    assert string in ('', 'S', 'P', 'SP', 'PS')


def _check_integration(result, error):
    """Naively check the integration error."""
    if abs(error / result) > 1e-7: warnings.warn(
        "The integration error is relatively big!")


def _printout(input_V, input_I_eff, output_V):
    print(
        "Noise signals:\n"
        f"Input RMS voltage: {round(1000 * input_V, 4)} mV.\n"
        "Effective input RMS current (\"ENI\"):"
        f" {round(1e9 * input_I_eff, 4)} nA.\n"
        f"Output RMS voltage: {round(1000 * output_V, 4)} mV.\n"
    )


import numpy as np
from matplotlib import pyplot as plt
import seaborn

seaborn.set()


def plot():
    Cs = np.linspace(1e-14, 100e-12, 1000)
    omegas = np.linspace(1e1, 1e3 + 100, 1000)
    noises = np.array([
        noise(lambda omega: Z_capacitance(omega, C), printout=False)
        for C in Cs
    ])
    densities = np.array([
        noise_density(omega, lambda omega: Z_capacitance(omega, C))
        for omega in omegas
    ])
    series_densities = np.array([
        noise_density(omega, lambda omega: Z_capacitance(omega, C), contributions='S')
        for omega in omegas
    ])
    parallel_densities = np.array([
        noise_density(omega, lambda omega: Z_capacitance(omega, C), contributions='P')
        for omega in omegas
    ])
    plt.plot(1e12 * Cs, 1000 * noises)
    plt.title('Theoretical Noise vs. Load Capacitance')
    plt.xlabel('Capacitance (pF)')
    plt.ylabel('Noise (mV)')
    plt.figure()
    plt.plot(omegas, densities, label='Combined')
    plt.plot(omegas, series_densities, label='Series')
    plt.plot(omegas, parallel_densities, label='Parallel')
    plt.legend()


if __name__ == '__main__':
    noise(lambda omega: Z_capacitance(omega, C))
    noise(lambda omega: Z_capacitance(omega, C_s), contributions='S')
    plot()
    plt.show()
