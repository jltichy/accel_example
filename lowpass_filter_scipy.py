#!/usr/bin/env python

import numpy as np # These are the packages necessary for the functions of interest.
from scipy.signal import butter, lfilter, freqz # Filters for the analysis.
import matplotlib # To be used for plotting data.
matplotlib.use('Agg') # Necessary for Cloud9.
import matplotlib.pyplot as plt

# The Butterworth filter is a type of signal processing filter designed to have 
# as flat a frequency response as possible in the passband. It is also referred 
# to as a maximally flat magnitude filter.

# Here is the original code - We will modify it below, which is why it is commented.
# def butter_lowpass(cutoff, fs, order=5): 
#     nyq = 0.5 * fs # The Nyquist Frequency is half the sampling rate.
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     # b is the numerator
#     # a is the denominator
#     return b, a
    
# # Let's try to change the order of the filter:
# def butter_lowpass(cutoff, fs, order=0.0000000010000000): 
#     nyq = 0.5 * fs # The Nyquist Frequency is half the sampling rate.
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     # b is the numerator
#     # a is the denominator
#     return b, a
# # It doesn't seem like this does anything.

# # Let's try to change the cutoff:
# def butter_lowpass(cutoff, fs, order=5): 
#     nyq = 0.5 * fs # The Nyquist Frequency is half the sampling rate.
#     normal_cutoff = cutoff/50 / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     # b is the numerator
#     # a is the denominator
#     return b, a
# # When we change the cutoff, it shifts the curve to the right or left.  If the 
# #cutoff is huge, then the plot gets shifted to the right and it the cutoff is 
# #really small, then the plot gets shifted to the left.  The line is hard to see
# #when the cutoff is very low.

# Let's go back to the original:
def butter_lowpass(cutoff, fs, order=5): 
    nyq = 0.5 * fs # The Nyquist Frequency is half the sampling rate.
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # b is the numerator
    # a is the denominator
    return b, a
    
# If we try to print b and a at this point, we can't, since b and a are simply
#returned as of now.  We'll need to call them and use them before they can be
#printed.

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
# Again, y is simply returned, so it cannot be printed yet.

# # Filter requirements.
# order = 6
# fs = 30.0       # sample rate, Hz
# cutoff = 3.667  # desired cutoff frequency of the filter, Hz

# # Let's try to change the order at this step:  Maybe it will make a difference.
# # Filter requirements.
# order = 1
# fs = 30.0       # sample rate, Hz
# cutoff = 3.667  # desired cutoff frequency of the filter, Hz
# # Yes, it makes a difference here.  The vectors for the numerator and denominator
# #are much larger when make a larger order and the vectors are much shorter when
# #the order is smaller.

# OK.  Let's go back to the original.
# Filter requirements.
order = 6  # When there is a larger order, then the curve will drop off at a 
#steeper rate.
fs = 30.0       # sample rate, Hz (Again, remember that Nyquist is half the
#sampling rate, so it would be 15 Hz. here.)
cutoff = 3.667  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Now we can print these because they have been called.
print b
print a
#b is the numerator and a is the denominator.

# Freqz will Compute the frequency response of a digital filter.
# w returns the normalized frequencies at which h was computed, in radians/sample.
# h is the frequency response.

# Plot the frequency response. The data is regularly sampled.
w, h = freqz(b, a, worN=8000) # This is used to generate the freq. response.
# Note that worN = 8000 is a random number used with freqz to create a smooth plot

plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko') # This puts a dot at the cutoff point.
plt.axvline(cutoff, color='k') # This draws a vertical line at the cutoff point.
plt.xlim(0, 0.5*fs) # Set the axis.
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()
plt.savefig('Lowpass Filter Frequency Response')
# This plot shows what the filter does, which is why it doesn't have a label
#on the y-axis.  For frequencies before the cutoff, the signal will be preserved
#but for frequencies after the cutoff, the signal will attenuate.

# Demonstrate the use of the filter.
# First make some data to be filtered.
T = 5.0         # seconds
n = int(T * fs) # total number of samples # I think int converts the value to integers.

print n
# n returns 150

# Let's create a vector of evenly spaced numbers over a specified interval.
# It will start at 0.
# It will stop at T, which is 5.0 in this case.
# It will have n values, which is 150 in this case.
# endpoint = False means that it will stop before the very end.
t = np.linspace(0, T, n, endpoint=False)
# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
# I think this data is just an example, but I don't really understand why
#we're adding three trig functions.  The 1.2, 9, and 12 affect the magnitudes.
#This is just sample data - it has multiple frequencies to give a good example.

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(data, cutoff, fs, order)

plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.ylabel('')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()
plt.savefig('Plot')
# OK.  Here's what we're seeing in this second plot.  BTW, the first plot shows
#what the filter will do.  Namely, at the cutoff, the signal will attenuate.  OK.
#The high frequency data (blue) gets attenuated and only the low frequency data 
#(green) is seen.  However, if we squint at the blue and green curves, we can see
#that there is a phase shift.  We should plot the phase to see what's going on.

# Let's try to plot the phase response in addition to the amplitude response,
#because that's all we have so far.
plt.clf()   # Clear figure
plt.close() # Close a figure window

angles = np.unwrap(np.angle(h))
plt.plot(w, angles, 'g')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [Radians]')
plt.savefig('Phase Plot')
# The previous figure (the one that showed what the filter would do, plus the blue
#and green curves) was just showing what would happen to the amplitude.  This figure
#shows what happens to the phase at different frequencies.

plt.clf()   # Clear figure
plt.close() # Close a figure window

# Let's do the example from the scipy documentation:
from scipy import signal # This is the signal processing toolbox
d, c = signal.butter(4, 100, 'low', analog=True)
# Remember that this butterworth filter returns the numerator (d) and denominator 
#(c) polynomials
w, h = signal.freqs(d, c)
# w (omega) returns the normalized frequencies at which h was computed, in radians/sample.
# h is the frequency response.
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
