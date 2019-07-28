"""Plot the data from the square pulse noise analysis."""

import seaborn
from matplotlib import pyplot

from read import read_file

seaborn.set()

def plot(readings):
    """Make simple plots of the readings for each channel."""
    fig, ax = pyplot.subplots()
    ax.set_title('Voltage Readings')
    ax.set_xlabel('Sample #')
    ax.set_ylabel('Reading (mV)')
    for channel, voltages in readings.items():
        pyplot.plot(voltages, label=f'Channel {channel}')
    pyplot.legend()

r = read_file(
    '../data/FAMP20_SquareWave_CalibrationLineTest_1MHz_0.5VGain_07.26'
    '.2019/FAMP20_6.0V_SquareWave_1MHz_0.5VGain_2019.07.26.16.52.txt',
    1 / 2
)
plot(r)
pyplot.show()
