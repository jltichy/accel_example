#!/usr/bin/env python

import numpy as np
from scipy.signal import butter, lfilter, freqz, detrend, decimate, welch
from scipy.integrate import cumtrapz
import math
import sys
import matplotlib
matplotlib.use('Agg') #This is required to run matplotlib on Google Chrome.
import matplotlib.pyplot as plt

# Here is some accelerometer data
f = open('acceldata','r')

# Read in the data
data = []
for line in f:
    data.append(int(line))

# The data is sampled at 100 Hz.  What is the time length of the file?
print('Here is the time span: ' + str(float(len(data))/100.) + ' seconds')

# Result - Here is the time span: 855.7 seconds

# Accelerometers output a voltage which is proportional to acceleration
# the acceleration is then digitized into counts

# If the digitizer is a 24 bit digitizer with a 40 Volt peak to peak range
# then the sensitivity is 2^24/40 which is counts/V  your data is in counts
# convert it to Volts

data = np.asarray(data,dtype=np.float32)/((2**24)/40.)

print(data)

# Here is the data in volts: 
#[-0.09621859 -0.0962162  -0.09622097 ..., -0.0949502  -0.09496927
# -0.09504557]

# Now that your data is in V you want to convert it to m/s^2
# The sensitivity of your accelerometer is 2g = 20 V what is 

data = data *2.*9.81/20.

print data

# Here is the data in m/s^2:
#[-0.09439044 -0.0943881  -0.09439278 ..., -0.09314615 -0.09316486
# -0.09323971]

# Now plot your data to see what you have
# Grab a time vector (t)

# Why are we dividing by 100? Think sample rate.  The data is sampled at 100Hz.
t=np.arange(len(data))/100.

fig = plt.figure()
plt.plot(t,data)
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s^2)')
plt.savefig('AccelExample1.jpg')
#plt.show()

# Now we want to low-pass the data at 10 Hz
# Define some parameters so we know what is going on
order = 2
fs = 100.0 #fs is the sampling rate, in Hz
corner = 10.

# If my sampling rate is 100 Hz what is my nyquist? Nyquist is half the
# sampling rate, so it's 50 Hz in this case
nyq = 0.5 * fs

# Look up this function. What kind of filter is this?
# Butterworth Filter - a signal processing filter that aims for  
#as flat a frequency response as possible in the passband. 

b, a = butter(order, corner/ nyq, btype='low', analog=False)

dataLP = lfilter(b,a,data)

# Now plot both the low-pass and the regular data

fig = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,data)
plt.subplot(2,1,2)
plt.plot(t,dataLP)
plt.savefig('AccelExample2.jpg')
#plt.show()
plt.close()

## START HERE ##

# Now change the corner to make this a 1 Hz low-pass how about a 0.1 Hz
# The spike at the front is called filter ringing and is annoying
# We can get rid of that by applying a taper

taper = np.hanning(len(data))

dataLP = lfilter(b,a,data*taper)

fig = plt.figure(1)
plt.plot(t,dataLP)
plt.savefig('AccelExample3.jpg')
#plt.show()
plt.close()

# Well now the data is even more ugly, but we can fix this by applying a high-pass
cornerHP = 0.01
b,a = butter(order, corner/nyq, btype='high', analog=False)
dataLP = lfilter(b,a,dataLP)

fig = plt.figure(1)
plt.plot(t,dataLP)
plt.savefig('AccelExample4.jpg')
#plt.show()
plt.close()

# What is the frequency content of dataLP?  Hint we let frequencies lower than 10 Hz make it through
# We also let frequencies of greater than 0.01 Hz make it through




####  Okay we should change focus  we have data in m/s  
####  what is the peak to peak acceleration
maxP = max(data)
minP = min(data)
print('The max peak is ' + str(maxP))
print('The min peak is ' + str(minP))

# Well that looks silly.  We should remove the linear trend because there is an off-set
data = detrend(data)

# Okay find the min and max again
maxP = max(data)
minP = min(data)
print('The max peak is ' + str(maxP))
print('The min peak is ' + str(minP))

# So the peak to peak would be
print('P to P: ' + str(abs(minP) + abs(maxP)))

# What about converting to m/s or m?
dataVelocity = cumtrapz(data,x=None, dx=0.01)
dataDisplacement = cumtrapz(dataVelocity,x=None, dx=0.01)

# Check what cumtrapz returns.  Notice we need a different time vector
tv = np.arange(len(dataVelocity))/100.
td = np.arange(len(dataDisplacement))/100.

fig = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(tv,dataVelocity)
plt.subplot(2,1,2)
plt.plot(td,dataDisplacement)
plt.savefig('AccelExample5.jpg')
#plt.show()
plt.close()

# Note most accelerometers don't integrate to displacement well because of noise

# What happens to the length and the Nyquist when we decimate by a factor of 10?
dataDec = decimate(data,10)
print('Here is dataDec len: ' + str(len(dataDec))) 

# Why do we need a new time vector?
t = np.arange(len(dataDec))/10.
fig = plt.figure(1)
plt.plot(t,dataDec)
plt.savefig('AccelExample6.jpg')
#plt.show()
plt.close()

# If I wanted to low-pass dataDec what would I need to change in my filter parameters?

# Now we want to low-pass the data at 1 Hz
# Define some parameters so we know what is going on
order = 2
fs = 10.0
corner = 1.

# If my sampling rate is 10 Hz what is my nyquist?  Notice we decimated the data
nyq = 0.5 * fs
# Look up this function what kind of filter is this?
b, a = butter(order, corner/ nyq, btype='low', analog=False)

dataLP = lfilter(b,a,dataDec)

fig = plt.figure(1)
plt.plot(t,dataLP)
plt.savefig('AccelExample7.jpg')
#plt.show()
plt.close()


#  Okay last part  Why don't we figure out what the PSD is?
# f is the frequency vector and P is the power
fs= 100.
f, P = welch(data, fs, nperseg = 512)

# The units of P are (m/s^2)^2 /Hz confusing
fig = plt.figure(1)
plt.plot(f,P)
plt.savefig('AccelExample8.jpg')
#plt.show()
plt.close()

# Yuck that is hart to read why not plot it on a log scale
fig = plt.figure(1)
plt.semilogx(f,P)
plt.savefig('AccelExample9.jpg')
#plt.show()
plt.close()

# Better, but we could convert it to dB 10*log10(P)
PdB = 10.*np.log10(P)
fig = plt.figure(1)
plt.semilogx(f,PdB)
plt.savefig('AccelExample10.jpg')
#plt.show()
plt.close()

# Okay, what if we compute the power for our Low-pass data
fLP, PLP = welch(dataLP,10., nperseg = 512)

fig = plt.figure(1)
plt.semilogx(f,PdB)
plt.semilogx(fLP, 10*np.log10(PLP))
plt.savefig('AccelExample11.jpg')
#plt.show()
plt.close()

# Which is PLP and which is PdB?  Hint one of them should not have as much power past 10Hz 
# Why do they have different frequencies?

# Okay one last exercise  What happens when we compute the PSD of our velocity trace?
fV, PV = welch(dataVelocity, 100., nperseg = 512)

# What are the units of PV?
fig = plt.figure(1)
plt.semilogx(fV, 10*np.log10(PV))
plt.semilogx(f,PdB)
plt.savefig('AccelExample12.jpg')
#plt.show()
plt.close()

# What happens if we multiply PV by omega^2=(2*pi*f)^2?
PA = 10.*np.log10(PV*(2*np.pi*fV)**2)

# What are the units of PA?  Why are these so similar?  
fig = plt.figure(1)
plt.semilogx(fV,PA)
plt.semilogx(f,PdB)
plt.savefig('AccelExample13.jpg')
plt.show()
plt.close()

# Take a look at 106 in the functional relationships
# https://en.wikipedia.org/wiki/Fourier_transform

# Why do we do (2*pi*f)^2 and not just (2*pi*f)?  Look at what Welch outputs
# This is a bit different from the FFFFT





