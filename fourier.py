"""Fourier analysis of experimental noise data."""

import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack

import read
import utils

# The voltage range spanned by the 2^14 possible counts from the
# digitizer. Can be 1/2 or 2 Volts.
digitizer_voltage_range = 1 / 2

# Spacing between voltage samples, in seconds.
sample_spacing = 2e-9

# Plotting.
_linewidth = 1
_dpi = 300

ryan = (
    r"../data"
    r"/FAMP20_NoiseData_7.19/FAMP20_6.0V"
)


def do_file(path):
    """Analyze the single file at `path`."""
    print(f"Doing file at {path}")
    readings = read.read(path, digitizer_voltage_range)

    fig, ax = plt.subplots()
    for channel, voltages in readings.items():
        _fourier(channel, voltages, ax)

    ax.set_title(f"Fast Fourier Transform:\n"
                 f"File {utils.file2name(path)}")
    ax.set_xlabel("Frequency ($10^8$ Hz)")
    ax.set_ylabel("Relative Amplitude")
    fig.legend()
    plt.show()

    return fig


def _fourier(channel, voltages, ax):
    """Fourier transform the `voltages` and plot them on `ax`."""
    fft_volts = fftpack.fft(voltages)

    # Convert index to frequency (based on my guess).
    assert len(fft_volts) == len(voltages)
    max_freq = 1 / (2 * sample_spacing)
    frequencies = np.linspace(0, max_freq, len(fft_volts))

    # Channel-dependent alphas.
    alpha = 0.1 if channel == 3 else 0.5

    ax.plot(
        frequencies[1:] / 1e8,  # 10^8 Hz
        fft_volts[1:],
        label=channel,
        linewidth=_linewidth,
        alpha=alpha
    )


if __name__ == '__main__':
    if len(sys.argv) > 1:
        datapaths = [file for file in sys.argv[1:] if 'decoded' not in file.lower()]
    else:
        ryan_files = [
            os.path.join(ryan, file)
            for file in next(os.walk(ryan))[2]
            if 'decoded' not in file.lower()
        ]
        datapaths = ryan_files
    for i, path in enumerate(datapaths):
        fig = do_file(path)

        # Simple save.
        # fig.savefig(str(i) + '.jpg', dpi=_dpi)
