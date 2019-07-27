"""
This is a work in progress that uses sympy to do the same calculations
as theory.py. It's not finished!

Theoretical calculations of amplifier noise signals.

Calculates electronic noise signal(s) for one center FCalPulse tube
segment. *Without* shaping from digitzer front-end amp.

TODO: Add preamp capacitance.

Written by Anson Kost, adapted from code written by
Prof. John Rutherfoord.

Variable names:
    Universal constants:
        k: Boltzmann constant.

    Circuit properties:
        A: Opamp amplification.

        rho: Cable characteristic impedance.
        T: Cable time delay.
        C: Detector capacitance.
        alpha: An angle that depends on the detector time constant (rho * C)
        (but not the cable length).

        R: Series noise source impedance.
        Q: Parallel noise source impedance.

        F2: Square magnitude of the series "transfer function"
        (noise to amp output).
        G2: Square magnitude of the parallel "transfer function."

        H2: Square magnitude of the frequency filter function.

    Other:
        omega: Angular frequency.
"""

import datetime
import sympy
from sympy.abc import k, A, rho, R, Q, C, T, alpha, omega
from scipy import integrate, inf

omega_cutoff = 1e10

print(datetime.datetime.now())

theta = omega * rho * C
alpha = sympy.acos((1 - theta ** 2) / (1 + theta ** 2))

F2 = (
        1 / (2 * (A + 1)) ** 2
        * (
                A ** 2
                + (A + 2) ** 2
                - A * (A + 2) * sympy.cos(2 * omega * T + alpha)
        )
)

H2 = sympy.Heaviside(omega_cutoff - omega)

symbolic_integrand = F2 * H2

inputs = {
        'A': 39,
        'rho': 50,
        'T': 4e-9,
        'C': 6e-12
}
numeric_integrand = sympy.lambdify(
        omega,
        symbolic_integrand.subs(inputs),
        modules=('numpy', 'sympy')
)
noise_V2 = integrate.quad(numeric_integrand, 0, omega_cutoff)

print(noise_V2)

print(datetime.datetime.now())
