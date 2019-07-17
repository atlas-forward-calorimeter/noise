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

digitizer_voltage_range = 1 / 2   # Input voltage range.
digitizer_sample_range = 2 ** 14  # Number of possible integer readings.
volts_per_sample = digitizer_voltage_range / digitizer_sample_range

# Default folder to analyze.
folder = (
    r"C:\Users\grape\Documents\aoi\fizz\FCal\Ryan"
    r"\FAMP20_NoiseData_7.15\FAMP20_6.0V"
)

# Some relevant filenames that we are keeping on hand.
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
        # Skip decoded data.
        if 'decodeddata' in file.lower():
            continue

        do_file(os.path.join(path, file))


def do_file(path):
    """Read in and analyze data from a file."""
    print(f"Doing file @ {path}.")

    # The "with" statement here guarantees that the file will be closed when
    # the statement finishes.
    with open(path) as file:

        # Loop over lines of the file.
        # Each line is a an integer from 0 to 255.
        for byte in file:
            if to_bits(byte).startswith('1010'):

                # The read_event function will advance our location in the
                # file some more (in addition to the for loop increment).
                read_event(file)


def read_event(file):
    """Read in the data from one event."""
    event_size = read_header(file)

    # 2 samples per word. Omit 4 header words.
    num_samples = 2 * (event_size - 4)

    readings, readings_squared = read_samples(
        file,
        num_samples=2 * (event_size - 4)
    )
    rms = math.sqrt(
            sum(readings_squared) / num_samples
            - (sum(readings) / num_samples) ** 2
    )

    print(f"RMS noise at the digitzer: {1000 * rms} mV.")


def read_header(file):
    """Read in one event from a file, assuming we are starting at the
    second header byte.
    """
    event_size_bytes = [to_byte(next(file)) for _ in range(3)]

    # Size of event, including header, as number of 4 byte words.
    event_size = (
            256 ** 2 * event_size_bytes[0]
            + 256 * event_size_bytes[1]
            + event_size_bytes[2]
    )

    # Check the board fail byte.
    assert to_bits(next(file))[5] == '0', "Hardware failure!"

    # Skip two lines and get the channel mask byte.
    _, _, ch_mask = (to_bits(next(file)) for _ in range(3))

    # For now, make sure channel 1 is the only one.
    assert ch_mask == '00000001', "More than one channel is used!"

    # Skip the last two words of the header.
    for _ in range(8):
        next(file)

    return event_size


def read_samples(file, num_samples):
    """Read in samples from one event."""
    readings, readings_squared = [], []

    for _ in range(num_samples):
        # Read the next two lines.
        high_byte, low_byte = (to_byte(next(file)) for _ in range(2))

        assert high_byte in range(24, 39), "Weird high byte!"
        
        value = 256 * high_byte + low_byte
        voltage = volts_per_sample * value

        readings.append(voltage)
        readings_squared.append(voltage ** 2)

    return readings, readings_squared


def to_byte(byte):
    """Make sure an object really represents an integer from 0 to 255,
    and return the integer.
    """
    byte = float(byte)
    assert byte.is_integer(), "Got a non-integer byte."

    byte = int(byte)
    assert byte >= 0, "Got a negative value for a byte."
    assert byte <= 255, "Got a byte value bigger than 255."

    return byte


def to_bits(byte):
    """Return a string representation of the bits in a byte.

    The Python `bin` function converts an integer byte into a
    string with a leading '0b' followed by the minimal amount of bits.
    This function cuts off the leading '0b' and fills in leading zeros
    until the result is 8 bits long.
    """
    return bin(to_byte(byte))[2:].zfill(8)


if __name__ == '__main__':
    # Only do this when this file is run by itself.

    # Get folder from command line argument.
    if len(sys.argv) > 1:
        folder = sys.argv[11]

    do_folder(folder)
