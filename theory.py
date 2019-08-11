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

import functools
import math
import warnings

import scipy.integrate
from scipy.constants import pi, Boltzmann, zero_Celsius

# Johnson-Nyquist noise sources.
R = 26  # Series noise source resistance.
Q = 770  # Parallel noise source resistance.

C = 6e-12  # Estimate of tube cals capacitance.
C_s = 5.7e-12  # Shower cals capacitance.

# Amplification from the preamp input to the feedback resistor.
# This represents the opamp, i.e. the "first stage" of amplification.
# The sign of A slightly affects the magnitude of the series transfer
# function `F` and thus the magnitude of the series noise at the preamp
# input.
A = -39

# Amplification from the preamp input to the output. The transimpedance
# is A2 * rho_amp (rho_amp is the input impedance). A2 represents the
# combination of both the "first stage" and "second stage" of
# amplification.
A2 = 87

T = 4e-9  # Cable time delay.
rho = 50  # Cable characteristic impedance.
rho_amp = rho  # FAMP input impedance.

omega_digitizer = 2 * pi * 250e6  # Digitizer high frequency cutoff.
tau_r = 1 / omega_digitizer  # Digitizer time constant.
tau_a = 0  # An unused frequency filter time constant.
C1, C2 = 1e-6, 1e-6  # Low frequency cutoff capacitances.

# Low frequency cutoff time constants.
tau_c1, tau_c2 = rho * C1, rho * C2

omega_lims = (1e3, 1 * omega_digitizer)  # omega integration limits.


def noise_comparison(Z_end_fns=None):
    """Calculate noise results with several impedances at the end of the
    cable. Also, for each impedance, integrate over both omega and
    ln(omega), using several integration limits for each of these two
    integration methods.

    Used for the August 1 results.
    """
    if not Z_end_fns:
        Z_end_fns = {
            'FCal': functools.partial(Z_capacitance, C=C),
            'SCal': functools.partial(Z_capacitance, C=C_s),
            'Open': Z_open,
            'Short': Z_short
        }
    limss_log = (
        # (1e-3, 1),
        # (1e-10, 1),
        # (1e-10, 500 * omega_digitizer),
        # (1, 1e5),
        (1e5, 500 * omega_digitizer),  # "Best" limits.
        # (1e5, 5000 * omega_digitizer),
        # (1e5, 1e10 * omega_digitizer),
        # (1e5, 1e30 * omega_digitizer),
    )
    limss = (
        # (1e-10, 500 * omega_digitizer),
        (1e5, 500 * omega_digitizer),  # "Best" limits.
        # (1e5, 5000 * omega_digitizer),
        # (1e5, 1e10 * omega_digitizer),
    )

    lines = [_printout_header]
    for label, Z_end_fn in Z_end_fns.items():
        for lims in limss_log:
            input_V = noise(Z_end_fn, omega_lims=lims)
            lines.append(_printout(input_V, label, lims))
        for lims in limss:
            input_V = noise(Z_end_fn, omega_lims=lims, log=False)
            lines.append(_printout(input_V, label, lims))
    print('\n'.join(lines))


_printout_header = '{:>25}, {:>25}, {:>40}'.format(
    'Output RMS voltage (mV)',
    'Input RMS voltage (uV)',
    'Effective input RMS current ("ENI") (nA)'
)
_printout_line_fmt = '{:25.5f}, {:25.4f}, {:40.3f}, {}, {}'.format
_printout_line_fmt2 = '{:25f}, {:25f}, {:40f}, {}, {}'.format


def _printout(input_V, label='', lims=''):
    output_V, eni = _input_V_calcs(input_V, A2, rho_amp)
    return _printout_line_fmt(
        1000 * output_V, 1e6 * input_V, 1e9 * eni, label, lims
    )


def _input_V_calcs(input_V, A2, rho_amp):
    """Get the effective preamp input noise current (ENI) and the output
    noise voltage from the noise voltage at the preamp input.

    :return: output voltage, effective input current (ENI)
    """
    output_V = A2 * input_V
    eni = input_V / rho_amp

    return output_V, eni


def noise(Z_end_fn=None, omega_lims=omega_lims, contributions='SP',
          printout=True, log=True):
    """The total integrated noise."""
    _check_contributions_string(contributions)

    if not Z_end_fn:
        def Z_end_fn(omega):
            return 0

    integration_result, error = 0, 0

    if log:
        lims = np.log(omega_lims)
        series_int = _series_integrand_log
        parallel_int = _parallel_integrand_log
    else:
        lims = omega_lims
        series_int = _series_integrand
        parallel_int = _parallel_integrand

    if 'S' in contributions:
        result, err = scipy.integrate.quad(
            series_int, lims[0], lims[1], args=(
                Z_end_fn, A, T, rho, rho_amp, tau_a, tau_r, tau_c1, tau_c2
            )
        )
        _check_integration(result, err)

        integration_result += _series_constant_factor * result

    if 'P' in contributions:
        result, err = scipy.integrate.quad(
            parallel_int, lims[0], lims[1], args=(
                Z_end_fn, T, rho, rho_amp, tau_a, tau_r, tau_c1, tau_c2
            )
        )
        _check_integration(result, err)

        integration_result += _parallel_constant_factor * result

    # Preamp input noise RMS voltage.
    input_V = _sqrt_constant_factor * math.sqrt(integration_result)

    return input_V


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


# Overall factors that are constant over the integration.
_constant_factor = (2 / pi) * Boltzmann * zero_Celsius
_sqrt_constant_factor = math.sqrt(_constant_factor)


def _integrand(omega, Z_end_fn, A, T, rho, rho_amp, tau_a, tau_r, tau_c1,
               tau_c2):
    """The sum of the series and parallel integrands with their
    individual constant factors. An overall constant factor is still
    excluded.

    This is not currently used.
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
    """Represents the series noise per omega without constant factors."""
    # |F|^2 |H|^2
    return abs(F(
        Z_cable(omega * T, Z_end_fn(omega), rho),
        A,
        rho_amp
    )) ** 2 * H_squared(omega, tau_a, tau_r, tau_c1, tau_c2)


def _series_integrand_log(ln_omega, Z_end_fn, A, T, rho, rho_amp, tau_a, tau_r,
                      tau_c1, tau_c2):
    """Represents the series noise per ln(omega) without constant
    factors.
    """
    # |F|^2 |H|^2
    omega = np.exp(ln_omega)
    return omega * abs(F(
        Z_cable(omega * T, Z_end_fn(omega), rho),
        A,
        rho_amp
    )) ** 2 * H_squared(omega, tau_a, tau_r, tau_c1, tau_c2)


_parallel_constant_factor = rho_amp ** 2 / Q


def _parallel_integrand(omega, Z_end_fn, T, rho, rho_amp, tau_a, tau_r, tau_c1,
                        tau_c2):
    """Represents the parallel noise per omega without constant factors."""
    # |G|^2 |H|^2
    return abs(G(
        Z_cable(omega * T, Z_end_fn(omega), rho),
        rho_amp
    )) ** 2 * H_squared(omega, tau_a, tau_r, tau_c1, tau_c2)


def _parallel_integrand_log(ln_omega, Z_end_fn, T, rho, rho_amp, tau_a, tau_r,
                            tau_c1, tau_c2):
    """Represents the parallel noise per ln(omega) without constant
    factors.
    """
    # |G|^2 |H|^2
    omega = np.exp(ln_omega)
    return omega * abs(G(
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
    tan_theta = np.tan(theta)
    return rho * (
            Z_end + 1j * rho * tan_theta
    ) / (rho + 1j * Z_end * tan_theta)


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
    return - 1j / (omega * C)


def Z_inductance(omega, L):
    return 1j * omega * L


def Z_open(omega):
    """ "Infinite" impedance for open circuits."""
    return 1e12j


def Z_short(omega):
    """This function exists because it has a descriptive name, but it
    just takes `omega` as an argument and returns 0.
    """
    return 0


def _check_contributions_string(string):
    """Check a string that represents which noise contributions to
    include in a calculation.
    """
    assert string in ('', 'S', 'P', 'SP', 'PS')


def _check_integration(result, error):
    """Naively check the integration error."""
    if result == 0:
        # Not possible to calculate the relative error of a null result.
        return
    rel_error = abs(error / result)
    if rel_error > 1e-7:
        msg = f"The relative integration error is big ({rel_error})!"
        warnings.warn(msg)

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
    noise_comparison()
    # noise(lambda omega: Z_capacitance(omega, C), omega_lims=(0, 3 * omega_digitizer), log=False)
    # noise(lambda omega: Z_capacitance(omega, C), omega_lims=(1e5, 20 * omega_digitizer))
    # noise(lambda omega: Z_capacitance(omega, C_s))
    # noise(lambda omega: 0)
    # noise(lambda omega: 1e17)
    # noise_log(lambda omega: Z_capacitance(omega, C))
    # noise_log(lambda omega: Z_capacitance(omega, C_s))
    # noise_log(lambda omega: 0)
    # noise_log(lambda omega: 1e17)
    # plot()
    # plt.show()
