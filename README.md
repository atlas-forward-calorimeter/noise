# noise
Theoretical and experimental calculations of amplifier noise signals.

## read.py
Reads in raw (not decoded) data from the digitizer. Here is an example:
```python
from read import read
readings = read('some/file/path', 1 / 2)
for channel, voltages in readings.items():
    # Do stuff with the `voltages` read on this `channel`.
```
In each list of voltages, the data from all events in the file is combined (this means there may be sudden jumps in voltages at the boundaries between events). Each voltage list is in sync with the others: `voltages[i]` corresponds to the same sample for all channels.

## fourier.py
Does some basic fast Fourier transforms. This is the file that we will want to add to.

### fourier._fourier(channel, voltages, ax)
This function uses SciPy's `fftpack` to perform an FFT on a list of voltages, assuming that the voltages are spaced at intervals given by `fourier.sample_spacing`. Then this function plots the results on `ax` and adds labels according for the `channel`.

### From The Command Line
Currently, fourier.py can be run from the command line in order to quickly perform FFTs on some data files. Here is an example:
```bash
$ python fourier.py FAMP20_9.1/*
```
This would perform FFTs for all the data files in the FAMP20_9.1/ folder. Any files that contain "decoded" in their name are skipped.

## utils.py
Utility functions. Just some functions that are useful in general. utils.noise_files iterates over files in a folder and skips any files that contain `decodeddata` in their names.

# Good Luck
