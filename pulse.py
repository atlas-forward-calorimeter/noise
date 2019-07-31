"""Plot the data from the square pulse noise analysis.

Units are in SI unless otherwise stated.

Notes:
    `np.asanyarray` is the same as `np.array` but leaves  NumPy arrays
    and subclasses of NumPy arrays untouched.
"""

import itertools

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from timeit import timeit

import read

seaborn.set()

# The offsets to test for matching the data from the two channels.
offsets = range(30)

# The voltage range spanned by the 2^14 possible counts from the
# digitizer. Can be 1/2 or 2 Volts.
digitizer_voltage_range = 1 / 2


# Number of indices to trim off when looking at the individual squares
# of square waves. Values are trimmed on either side in order to avoid
# the transition period between squares.
square_cutoff = 25

# Keep the path to the data handy.
datapath = (
    '../data/FAMP20_SquareWave_CalibrationLineTest_1MHz_0.5VGain_07.26'
    '.2019/FAMP20_6.0V_SquareWave_1MHz_0.5VGain_2019.07.26.16.52.txt'
)


def go(filepath):
    """Read in pulsed data and analyze the pulses, one by one."""
    readings = {
        # Voltages are relative to the mean over the channel.
        channel: np.asanyarray(voltages) - np.mean(voltages)
        for channel, voltages
        in read.read(filepath, digitizer_voltage_range).items()
    }
    assert readings.keys() == {0, 1}, \
        f"Unrecognized keys: {readings.keys()}"
    crossings = {
        channel: _crossings(voltages) for channel, voltages in readings.items()
    }
    offset = _match_offset(
        guide=crossings[1], to_match=crossings[0], offsets=offsets
    )
    analyses = [
        _crunch_square(i, low, high, volts0, volts1)
        for i, ((low, high), volts0, volts1)
        in enumerate(_sections(
            crossings[1], readings[0][offset:], readings[1][:-offset]
        ))
    ]


def plot(volts0, volts1, title=None, save=None):
    """Make a simple plot of the voltage readings on both channels."""
    volts0, volts1 = np.asanyarray(volts0), np.asanyarray(volts1)
    title = title or 'Noise readings on both channels.'
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Reading (mV)')
    plt.plot(1000 * volts0, label=f'Channel 0')
    plt.plot(1000 * volts1, label=f'Channel 1')
    plt.legend(loc='upper right')
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')


def _crunch_square(i, low, high, volts0, volts1):
    """Analyze one flat "edge" of a square from some square wave noise
    readings.

    See `_sections` for documentation of the function parameters.
    """
    # Trim off the transition period between squares of the square wave.
    square = volts0[square_cutoff:-square_cutoff]
    mean, std = square.mean(), square.std()

    message = (
        f'Noise for "square" {i}. Samples {low} to {high}.\n'
        f'{1000 * mean:.2f} mV mean and {1000 * std:.2f} mV std on '
        f'channel 0.'
    )
    print(message)

    plot(volts0, volts1, message, save=str(i) + '.jpg')
    plt.show()


def _match_offset(guide, to_match, offsets, tol=0):
    """
    Find the effective offset of the values in `to_match` relative to
    those in `guide`.

    For each offset in `offsets` and for each value in `guide`, this
    function attempts to find a value in `to_match` that, when shifted
    by the negative of the offset being tested, matches the guide value
    within the allowed tolerance. If at least one such value is found, a
    match is recorded for the offset. The offset that produces the most
    matches is returned.

    :param guide: The values to attempt to match values in `to_match`
        to.
    :param to_match: The values to match to the guide values.
    :param tol: Tolerance for the difference in matched values. This
        should be smaller than the half separations between guide values
        in order to avoid matching a single value in `to_match` to two
        or more guide values.
    :return: The effective offset of `to_match` relative to `guide`.
    """
    matches = {
        # offset: number of matches
        offset: sum(
            (np.absolute(to_match - offset - guide_value) <= tol).any()
            for guide_value in guide
        )
        for offset in offsets
    }
    return max(matches, key=matches.get)


def _crossings(values):
    """Find the points where an array of values crosses its mean. If the
    values represent a square wave, this finds the edges of the squares.
    """
    values = np.asanyarray(values)
    relative = values - values.mean()
    return np.where(np.diff(np.sign(relative)))[0]


def _sections(boundaries, *valuess):
    """
    Iterate over sections of sets of values.

    If the sets of values in `valuess` are not all the same length, then
    this function acts like `zip` and iterates until it reaches the end
    of the shortest set of values.

    :param values: The sets of values to iterate over in sections.
    :param boundaries: Should contain the indices of the final value of
        each section.
    :return: The boundaries of the sections and the sections of values
        themselves.
    """
    boundaries = np.asanyarray(boundaries)
    boundaries = boundaries + 1
    for low, high in zip(
            itertools.chain([0], boundaries),
            itertools.chain(boundaries, [len(valuess[0])])
    ):
        yield ((low, high), *(vals[low:high] for vals in valuess))


if __name__ == '__main__':
    r = read.read(datapath, digitizer_voltage_range)
    plot(r[0], r[1], save='everything.jpg')
    plt.show()

    go(datapath)

    # TODO: Throw this all in a function, or remove it.
    # r = read(
    #     '../data/FAMP20_SquareWave_CalibrationLineTest_1MHz_0.5VGain_07.26'
    #     '.2019/FAMP20_6.0V_SquareWave_1MHz_0.5VGain_2019.07.26.16.52.txt',
    #     1 / 2
    # )
    # r = {channel: np.asanyarray(readings) for channel, readings in r.items()}
    # r = {
    #     channel: readings - readings.mean() for channel, readings in r.items()
    # }
    # cross0, cross1 = (_crossings(read) for read in r.values())
    # offset = _match_offset(cross1, cross0, offsets)
    #
    # for (lo, hi), square0, square1 in _sections(cross1 + offset, r[0], r[1]):
    #     print(lo, hi)
    #     plot({0: square0, 1: square1})
    #     plt.show()
    #
    # plot(r)
    # plot({'0': r[0][offset:], '1': r[1][:-offset]})
    # # plt.show()
