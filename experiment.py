"""Read in and analyze raw noise data from the digitizer.

This script is minimalistic, so a reader can understand it without
knowing a lot of Python!

The digitzer is a Caen DT5730 (user manual UM3148). The data consists of
events which each begin with a header of four four-byte words followed
by event samples, the number of which is specified in the header.

Written by Anson Kost. July 2019.
"""

# Import modules from the Python Standard Library.
import os

# What file to read in.
folder = r"C:\Users\grape\Documents\aoi\fizz\FCal\Ryan\FAMP20_NoiseData_7.15\FAMP20_6.0V"
filename = r"FAMP20_Ch1_FCal_Disconnected_2019.07.15.14.33.txt"
path = os.path.join(folder, filename)


def read_event(file):
    """Read in the data from one event."""
    event_size = read_header(file)
    num_samples = 2 * (event_size - 4)

    for _ in range(num_samples):
        high_byte, low_byte = (to_byte(file.readline()) for _ in range(2))

        assert high_byte in range(27, 36), f"Weird high byte: {high_byte}!"

        value = 256 * high_byte + low_byte
        print(value)


def read_header(file):
    """Read in one event from a file, assuming we are starting at the
    second header byte.
    """
    event_size_bytes = [to_byte(file.readline()) for _ in range(3)]

    # Size of event, including header, as number of 4 byte words.
    event_size = (
            256 ** 2 * event_size_bytes[0]
            + 256 * event_size_bytes[1]
            + event_size_bytes[2]
    )

    # Check the board fail byte.
    assert to_bits(file.readline())[5] == '0', "Hardware failure!"

    # Skip two lines and get the channel mask byte.
    _, _, ch_mask = (to_bits(file.readline()) for _ in range(3))

    # For now, make sure channel 1 is the only one.
    assert ch_mask == '00000001', "More than one channel is used!"

    # Skip the last two words of the header.
    for _ in range(8):
        file.readline()

    return event_size


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


# The "with" statement here guarantees that the file will be closed when
# the statement finishes.
with open(path) as file:

    # Loop over lines of the file.
    # Each line is a an integer from 0 to 255.
    for byte in file:
        if to_bits(byte)[:4] == '1010':
            read_event(file)
