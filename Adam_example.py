#!/usr/bin/env python

import numpy as np

from scipy.signal import butter, lfilter, freqz

import matplotlib.pyplot as plt
cutoff = 1.


fs = 40.
nyq = 0.5*fs

normal_cutoff = cutoff / nyq


b, a = butter(4, normal_cutoff, btype='low', analog=False)
print(b)
print(a)


# Here is the data going into the system

# So we have data being sampled at 40 Hz
t = np.arange(0,1000)/fs


data = 5.*np.sin(2.*np.pi*0.1*t)

# Apply our filter to the data
dataout = lfilter(b, a, data)

# Here is the frequency response
w,h = freqz(b ,a, worN=3000)

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
