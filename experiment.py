"""Read in and analyze raw noise data from the digitizer.

This script is minimalistic, so a reader can understand it without
knowing a lot of Python! Still, there are enough checks for a user to be
"fairly confident" in the results.

The digitzer is a Caen DT5730 (user manual UM3148). The data consists of
events which each begin with a header of four four-byte words followed
by event samples, the number of which is specified in the header. All
units are SI unless otherwise stated.

Written by Anson Kost. July 2019.
"""

# Import modules from the Python Standard Library.
import math
import os
import sys

# The NumPy module, from Anaconda!
import numpy

# Tell NumPy to print numbers to four significant figures.
numpy.set_printoptions(suppress=True, precision=4)

digitizer_voltage_range = 1 / 2   # Input voltage range.
digitizer_sample_range = 2 ** 14  # Number of possible integer readings.
volts_per_sample = digitizer_voltage_range / digitizer_sample_range

# Default folder to analyze.
folder = 'data/FAMP20_6V_MultiCh'
folder2 = 'data/FAMP20_5.5V_MultiCh'
folder3 = 'data/6V'

# Some relevant filenames to keep on hand.
filename = 'FAMP20_6V_MultiCh_Ch0_FCal_Connected_2019.07.17.14.34.txt'
path = os.path.join(folder, filename)
filenames = [
    r"FAMP20_Ch1_FCal_Disconnected_2019.07.15.14.33.txt",
    r"FAMP20_Ch10_FCal_Disconnected_2019.07.15.14.55.txt",
    r"FAMP20_Ch20_Cable_Shorted_2019.07.15.15.05.txt",
    r"FAMP20_Ch1_FCal_Connected_2019.07.15.14.31.txt"
]


def do_folder(path):
    """Read in and analyze all files in a folder."""
    path, dirs, files = next(os.walk(path))
    for file in files:
        # Skip decoded files.
        if 'decodeddata' in file.lower():
            continue

        do_file(os.path.join(path, file))


def do_file(path):
    """Read in events from a file, combine them together, and analyze
    them.
    """
    path = os.path.normpath(path)  # "Normalize" the path.

    file_readings = None

    # The `with` statement here guarantees that the file will be closed when
    # the statement finishes.
    with open(path) as file:
        # Loop over lines of the file.
        # Each line is an integer from 0 to 255.
        for byte in file:
            if to_bits(byte).startswith('1010'):
                # Assume the first byte that starts with '1010' is the
                # start of a header, and read in the event.
                # The read_event function will advance our location in
                # the file past the event, and the for loop will pick
                # up there.
                event_readings = read_event(file)

                if not file_readings:
                    file_readings = event_readings
                else:
                    # Update the file readings with this event's
                    # readings.
                    for channel, readings in event_readings.items():
                        file_readings[channel] += readings

    print(f"Analysis of {path}.")
    analyze(file_readings)


def analyze(readings):
    """Do calculations on a set of readings and print the results."""
    channels = list(readings.keys())
    readings_numpy = [numpy.array(reads) for reads in readings.values()]

    num_samples = len(readings_numpy[0])
    means = [reads.mean() for reads in readings_numpy]
    rms = numpy.array([
        math.sqrt((reads ** 2).mean() - mean ** 2)
        for reads, mean in zip(readings_numpy, means)
    ])
    correlations = numpy.array([
        [
            ((reads_i - mean_i) * (reads_j - mean_j)).mean() / (rms_i * rms_j)
            for reads_j, mean_j, rms_j in zip(readings_numpy, means, rms)
        ]
        for reads_i, mean_i, rms_i in zip(readings_numpy, means, rms)
    ])

    print(
        f"{num_samples} samples."
        f"\nChannels: {channels}"
        f"\nNoise (mV): {1000 * rms}"
        "\nPearson's correlation coefficients:"
        f"\n{correlations}"
    )


def read_event(file):
    """Read in the data from one event."""
    channels, words_per_channel = read_header(file)

    return {
        channel: read_samples(file, words_per_channel)
        for channel in channels
    }


def read_header(file):
    """Read in one event from a file, assuming we are starting at the
    second header byte.
    """
    # The next three bytes are for the event size.
    event_size_bytes = [to_byte(next(file)) for _ in range(3)]

    # The event size includes the header and is in units of 4-byte words.
    event_size = (
            256 ** 2 * event_size_bytes[0]
            + 256 * event_size_bytes[1]
            + event_size_bytes[2]
    )

    # Check the board fail byte.
    assert to_bits(next(file))[5] == '0', "Hardware failure!"

    # Skip two bytes and then get the channel mask byte.
    _, _, ch_mask = (to_bits(next(file)) for _ in range(3))

    # Convert the channel mask to a list of channel numbers.
    # The [::-1] after ch_mask reverses the order of its characters,
    # and enumerate is a built-in function which will loop through
    # the characters and return the loop count along with each
    # character.
    # By the way, this is a normal list comprehension, but the `if`
    # syntax at the end serves to filter out which elements are added to
    # the new list.
    channels = [
        position
        for position, bit in enumerate(ch_mask[::-1])
        if bit == '1'
    ]

    # Number of sample words per channel. The 4 header words are omitted.
    words_per_channel = (event_size - 4) / len(channels)
    assert words_per_channel.is_integer(), (
        "The total number of sample words in this event is not a"
        " multiple of the number of channels!"
    )
    words_per_channel = int(words_per_channel)

    # Check the channels.
    assert ch_mask in ('00000001', '00001111'), \
        f"Bad channel mask: {ch_mask}!"

    # Skip the last two words of the header.
    for _ in range(8):
        next(file)

    return channels, words_per_channel


def read_samples(file, num_words):
    """Read in a given number of words."""
    readings = []

    for _ in range(2 * num_words):
        # Read in the next two lines/bytes.
        high_byte, low_byte = (to_byte(next(file)) for _ in range(2))

        assert high_byte in range(22, 40), \
            f"Weird high byte: {high_byte}!"

        if high_byte not in range(29, 34):
            pass  # continue

        value = 256 * high_byte + low_byte
        voltage = volts_per_sample * value

        readings.append(voltage)

    return readings


def to_byte(byte):
    """Make sure an object really represents an integer from 0 to 255,
    and return the integer.
    """
    byte = float(byte)
    assert byte.is_integer(), f"Got a non-integer byte: {byte}!"

    byte = int(byte)
    assert byte >= 0, f"Got a negative value for a byte: {byte}!"
    assert byte <= 255, f"Got a byte value bigger than 255: {byte}!"

    return byte


def to_bits(byte):
    """Return a string representation of the bits in a byte.

    `bin` is a Python built-in function which converts an integer byte
    into a string representation. The string is prefixed by '0b' and
    uses the minimal number of bits so that there are no leading zeros.
    This function cuts off the leading '0b' and fills in leading zeros
    until the result is 8 bits long.
    """
    return bin(to_byte(byte))[2:].zfill(8)


if __name__ == '__main__':
    # Only do the following when this script is run directly.

    if len(sys.argv) > 1:
        # Get folder from command line argument.
        folder = sys.argv[1]

    do_folder(folder2)
