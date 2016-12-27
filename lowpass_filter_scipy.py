#!/usr/bin/env python

import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib
matplotlib.use('Agg') # Necessary for Cloud9
import matplotlib.pyplot as plt

# The Butterworth filter is a type of signal processing filter designed to have 
# as flat a frequency response as possible in the passband. It is also referred 
# to as a maximally flat magnitude filter.

def butter_lowpass(cutoff, fs, order=5): 
    nyq = 0.5 * fs # The Nyquist Frequency is half the sampling rate.
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # b is the numerator
    # a is the denominator
    return b, a

# This doesn't seem to be working.  Let's try it later.
#print "Numerator: %d" % b
#print "Denominator: %d" % a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# I don't know why this isn't working...
#print b
#print a 

# Filter requirements.
order = 6
fs = 30.0       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response. The data is regularly sampled.
w, h = freqz(b, a, worN=8000) # This is used to generate the freq. response.
# Note that worN = 8000 is a random number used with freqz to create a smooth
# plot
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


# Demonstrate the use of the filter.
# First make some data to be filtered.
T = 5.0         # seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)
# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(data, cutoff, fs, order)

plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()
plt.savefig('Plot')

plt.clf()   # Clear figure
plt.close() # Close a figure window

# Let's do the example from the scipy documentation:
from scipy import signal
d, c = signal.butter(4, 100, 'low', analog=True)
w, h = signal.freqs(d, c)
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()
plt.savefig('Plot2')
