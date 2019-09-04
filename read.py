"""Read in data from the digitizer."""

from collections import defaultdict

# Digitizer resolution. 2^14 possible count values.
count_range = 2 ** 14


def read(path, voltage_range):
    """Read in events from a file, combine them, and return them."""
    readings = defaultdict(list)
    # The `with` statement here guarantees that the file will be closed
    # when the statement finishes. This is a standard way to open files.
    with open(path) as file:
        # The following loops over lines of the file.
        # Each line is an integer from 0 to 255.
        for byte in file:
            if _to_bits(byte).startswith('1010'):
                # Assume the first byte that starts with '1010' is the
                # start of a header, and read in the event.
                # The read_event function will advance our location in
                # the file past the event, and the for loop will pick
                # up there.
                event_readings = _read_event(file,
                                             scale=voltage_range / count_range)
                # Update the file readings with this event's
                # readings.
                for channel, voltages in event_readings.items():
                    readings[channel] += voltages
    return readings


def _read_event(file, scale):
    """Read in the data from one event."""
    channels, words_per_channel = _read_header(file)
    return {
        channel: _read_samples(file, words_per_channel, scale)
        for channel in channels
    }


def _read_header(file):
    """Read in one event from a file, assuming we are starting at the
    second header byte.
    """
    # The next three bytes are for the event size. The event size
    # includes the header and is in units of 4-byte words.
    event_size = (
            256 ** 2 * _to_byte(next(file))
            + 256 * _to_byte(next(file))
            + _to_byte(next(file))
    )

    # Check the "board fail" byte.
    assert _to_bits(next(file))[5] == '0', "Hardware failure!"

    # Skip two bytes and then get the channel mask byte.
    next(file)
    next(file)
    ch_mask = _to_bits(next(file))

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

    # Number of sample words per channel. The 4 header words of the
    # event are not included.
    words_per_channel = (event_size - 4) / len(channels)
    assert words_per_channel.is_integer(), (
        "The total number of sample words in this event is not a"
        " multiple of the number of channels!"
    )
    words_per_channel = int(words_per_channel)

    # Check the channels.
    assert ch_mask in ('00000011', '00001111'), \
        f"Bad channel mask: {ch_mask}!"

    # Skip the last two words of the header.
    for _ in range(8):
        next(file)

    return channels, words_per_channel


def _read_samples(file, num_words, scale):
    """Read in a given number of words and return scaled values."""
    readings = []
    for _ in range(num_words):
        second_sample, first_sample = _read_sample(file), _read_sample(file)
        if scale != 1:
            first_sample *= scale
            second_sample *= scale
        readings.append(first_sample)
        readings.append(second_sample)
    return readings


def _read_sample(file):
    """Read in and check a single sample consisting of two bytes, and
    return the value in counts."""
    high_byte, low_byte = _to_byte(next(file)), _to_byte(next(file))

    assert high_byte in range(13, 56), \
        f"Weird high byte: {high_byte}! With low byte {low_byte}."
    if high_byte not in range(29, 34):
        # Using `continue` here will skip samples outside of this
        # range.
        pass  # continue

    return 256 * high_byte + low_byte


def _to_byte(byte):
    """Make sure an object really represents an integer from 0 to 255,
    and return the integer.
    """
    byte = float(byte)
    assert byte.is_integer(), f"Got a non-integer byte: {byte}!"

    byte = int(byte)
    assert byte >= 0, f"Got a negative value for a byte: {byte}!"
    assert byte <= 255, f"Got a byte value bigger than 255: {byte}!"

    return byte


def _to_bits(byte):
    """Return a string representation of the bits in a byte.

    `bin` is a Python built-in function which converts an integer byte
    into a string representation. The string is prefixed by '0b' and
    uses the minimal number of bits so that there are no leading zeros.
    This function cuts off the leading '0b' and fills in leading zeros
    until the result is 8 bits long.
    """
    return bin(_to_byte(byte))[2:].zfill(8)


if __name__ == '__main__':
    # Testing the reader.
    r = read(
        '../data/FAMP20_SquareWave_CalibrationLineTest_1MHz_0.5VGain_07.26'
        '.2019/FAMP20_6.0V_SquareWave_1MHz_0.5VGain_2019.07.26.16.52.txt',
        1 / 2
    )

    import math

    for k, v in r.items():
        m = sum(v) / len(v)
        s = math.sqrt(sum([vv ** 2 for vv in v]) / len(v) - m ** 2)
        print('ch', k, m, s, v[:5])
