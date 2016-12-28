#!/usr/bin/env python

#We make a filter, have some data going into the system, and have some data 
#going out of the system.  By changing around the different filter parameters,
#you have different inputs to outputs.

import numpy as np

from scipy.signal import butter, lfilter, freqz
import matplotlib
matplotlib.use('Agg') # Necessary for Cloud9
import matplotlib.pyplot as plt

cutoff = 1.
# Why is this written as 1. and not just 1
#??

fs = 40. # This is the sampling frequency.
nyq = 0.5*fs # Nyquist is half the sampling frequency, so in this case it is 20 Hz.
 
normal_cutoff = cutoff / nyq # This will be our corner cutoff.

b, a = butter(4, normal_cutoff, btype='low', analog=False)
print(b)
print(a)
# b is the numerator polynomial and a is the denominator polynomial.

# Here is the data going into the system.

# So we have data being sampled at 40 Hz
t = np.arange(0,1000)/fs

data = 5.*np.sin(2.*np.pi*0.1*t)

# Apply our filter to the data
dataout = lfilter(b, a, data)

# Here is the frequency response
w,h = freqz(b ,a, worN=3000)
# Remember that worN smooths the data.

fig = plt.figure(1)
plt.subplot(3,1,1)
plt.plot(t,data,label='Data In')
plt.plot(t,dataout,label='Data Out')
plt.legend()
plt.subplot(3,1,2)
plt.plot(0.5*fs*w*np.pi, np.abs(h),label='Amplitude Response')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.xlim(0, 0.5*fs)
plt.subplot(3,1,3)
plt.plot(0.5*fs*w*np.pi, (180./np.pi)*np.unwrap(np.angle(h)),label='Phase Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.xlim(0,0.5*fs)
plt.show()
plt.savefig('Adam - Plot')

# After going through the accelstuff.py plus the lowpass_filter_scipy.py
#documents, I feel like I have a pretty good handle of what's going on in this
#script.