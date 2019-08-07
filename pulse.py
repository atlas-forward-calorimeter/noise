"""Plot the data from the square pulse noise analysis.

Units are in SI unless otherwise stated.

Notes:
    `np.asanyarray` is the same as `np.array` but leaves  NumPy arrays
    and subclasses of NumPy arrays untouched.
"""

import itertools
import os
import sys

import numpy as np
import seaborn
from matplotlib import pyplot as plt

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
rise_fall_cutoff = 25

# Plot parameters.
_linewidth = 1

# Format and dots-per-inch of saved images.
_image_format = 'jpg'
_image_dpi = 300

# Keep the path to the data handy.
# datapath = (
#     '../data'
#     '/FAMP20_SquareWave_CalibrationLineTest_1MHz_0.5VGain_07.26.2019'
#     '/FAMP20_6.0V_SquareWave_1MHz_0.5VGain_2019.07.26.16.52.txt'
# )
# datapath = (
#     '../data'
#     '/FAMP20_SquareWave_CalibrationLineTest_1MHz_08.05.2019'
#     '/FAMP20_SqW_1MHz_FeCore_PowerIso_FaradayCage_2019.08.05.15.47.txt'
# )
# datapath = (
#     '../data'
#     '/FAMP20_SquareWave_CalibrationLineTest_1MHz_08.05.2019'
#     '/FAMP20_SqW_1MHz_FeCore_PowerIso_FaradayCage_2019.08.05.16.09.txt'
# )
# datapath = (
#     '../data'
#     '/FAMP20_SquareWave_CalibrationLineTest_1MHz_08.06.2019'
#     '/FAMP20_SqW_1MHz_FeCore_PowerIso_FdyCage_CryoCan_2019.08.06.13.12'
#     '.txt'
# )
datapath = (
    '../data'
    '/FAMP20_SquareWave_CalibrationLineTest_1MHz_08.06.2019'
    '/FAMP20_SqW_1MHz_FeCore_PowerIso_FdyCage_CryoCan_2019.08.06.13.13'
    '.txt'
)

# Title for the particular set of data being analyzed.
title = 'Faraday Cage and Cryostat. File 2. 8-06.'

# Where to save the analysis results.
save_dir = 'cage-cryo-8-06/file2'


def do_folders(
        folders, max_squares=None, save_dir=None, show=False, title=None
):
    """Analyze several folders of data and save the results."""
    for folder in folders:
        assert os.path.isdir(folder), "This isn't a folder."
        root, dirs, files = next(os.walk(folder))
        assert root == folder

        if save_dir:
            folder_save_dir = os.path.join(save_dir, _dir2name(folder))
        else:
            folder_save_dir = None

        for file in files:
            if 'decoded' in file.lower():
                continue

            if folder_save_dir:
                file_save_dir = os.path.join(folder_save_dir, _file2name(file))
            else:
                file_save_dir = None

            go(os.path.join(folder, file), max_squares, save_dir, show, title)


def go(filepath, max_squares=None, save_dir=None, show=False, title=None):
    """Read in pulsed data and analyze the pulses, one by one.

    :param readings: Digitizer readings.
    :param max_squares: Maximum number of squares to "crunch."
    :param save_dir: Folder to save the plots in.
    :param show: If `True`, display the plots one by one.
    :param title: Overall plot title.
    :return:"""
    if save_dir:
        os.makedirs(save_dir)

    readings = {
        # Voltages are relative to the mean over the channel.
        channel: np.asanyarray(voltages) - np.mean(voltages)
        for channel, voltages
        in read.read(filepath, digitizer_voltage_range).items()
    }
    plot(readings[0], readings[1])
    plt.show()

    crunch_squares(readings, max_squares, save_dir, show, title)


def plot(volts0, volts1, title=None, save=None):
    """Make a simple plot of the voltage readings on both channels."""
    volts0, volts1 = np.asanyarray(volts0), np.asanyarray(volts1)
    title = title or 'Noise readings on both channels.'
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Reading (mV)')
    plt.plot(1000 * volts0, label=f'Channel 0', linewidth=_linewidth)
    plt.plot(1000 * volts1, label=f'Channel 1', linewidth=_linewidth)
    plt.legend(loc='upper right')
    if save:
        plt.savefig(save, dpi=_image_dpi, bbox_inches='tight')
    return fig


def crunch_squares(
        readings, max_squares=None, save_dir=None, show=False, title=None
):
    """Analyze square pulses one by one.

    See `go` for documentation of this function's parameters.
    """
    _check_readings(readings)
    crossings = {
        channel: _crossings(voltages) for channel, voltages in readings.items()
    }
    offset = _match_offset(
        guide=crossings[1], to_match=crossings[0], offsets=offsets
    )
    print(f'Offset: {offset} samples.')

    max_squares = max_squares or len(crossings[1]) + 1
    analyses = [
        _crunch_square(
            i, low, high, volts0, volts1, save_dir=save_dir, show=show, title=title
        )
        for i, ((low, high), volts0, volts1)
        in zip(
            range(1, max_squares + 1),
            _sections(
                crossings[1], readings[0][offset:], readings[1][:-offset]
            )
        )
    ]


def _crunch_square(i, low, high, volts0, volts1, save_dir, show, title=None):
    """Analyze one flat "edge" of a square from some square wave noise
    readings.
    """
    # Trim off the transition period between squares of the square wave.
    square = volts0[rise_fall_cutoff:-rise_fall_cutoff]
    mean, std = square.mean(), square.std()

    message = (
        f'Square {i}. Samples {low} to {high}.\n'
        f'{1000 * mean:.2f} mV mean and {1000 * std:.2f} mV std on ch0.'
    )
    print(message)

    if save_dir:
        filename = '.'.join((str(i), _image_format))
        savepath = os.path.join(save_dir, filename)
    else:
        savepath = None

    plot_title = '\n'.join((title, message)) if title else message
    fig = plot(volts0, volts1, plot_title, save=savepath)
    if show:
        plt.show()
    plt.close(fig)


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


def _check_readings(readings):
    """Make sure the readings from the digitizer are in a form we
    expect.
    """
    assert readings.keys() == {0, 1}, \
        f"Unrecognized keys: {readings.keys()}"


def _file2name(file_path):
    """
    Return the name of the file at `file_path` without its extension.

    :param file_path: File path.
    :type file_path: str
    :return: Nice name.
    :rtype: str
    """
    tail, head = os.path.split(file_path)
    assert head != '', "Is this a directory instead of a file_path?"

    return head.split('.')[0]


def _dir2name(dir_path):
    """
    Return the name of the directory at `dir_path`.

    :param dir_path: Directory path.
    :type dir_path: str
    :return: Nice name.
    :rtype: str
    """
    tail, head = os.path.split(dir_path)
    if head == '':
        tail, head = os.path.split(tail)

    return head


if __name__ == '__main__':
    # do_folders(sys.argv[1:], max_squares=25, save_dir='analysis')
    go(datapath, max_squares=25, save_dir=save_dir, show=False, title=title)
