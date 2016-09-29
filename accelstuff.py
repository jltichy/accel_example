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

# Now change the corner to make this a 1 Hz low-pass how about a 0.1 Hz
# The spike at the front is called filter ringing and is annoying
# We can get rid of that by applying a taper

corner2 = 0.1
b, a = butter(order, corner2/ nyq, btype='low', analog=False)
dataLP = lfilter(b,a,data)
fig = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,data)
plt.subplot(2,1,2)
plt.plot(t,dataLP)
plt.savefig('AccelExample2a.jpg')
#plt.show()
plt.close()

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

#Why didn't the above code use cornerHP?  Let's try it here:
b,a = butter(order, cornerHP/nyq, btype='high', analog=False)
dataLP = lfilter(b,a,dataLP)
fig = plt.figure(1)
plt.plot(t,dataLP)
plt.savefig('AccelExample4a.jpg')
#plt.show()
plt.close()

#There is a difference between plots 4 and 4a.

# What is the frequency content of dataLP?  
# Hint we let frequencies lower than 10 Hz make it through
# We also let frequencies of greater than 0.01 Hz make it through

# THis section is wrong - these are the peaks, not the frequency content.
dataLP_max = np.amax(dataLP)
print dataLP_max
# output = 2.10013666896e-06
dataLP_min = np.amin(dataLP)
print dataLP_min
# output = -2.60303208937e-06

####  Okay we should change focus. We have data in m/s.

## Question - I thought the units were m/s^2.

####  what is the peak to peak acceleration?
maxP = max(data)
minP = min(data)
print('The max peak is ' + str(maxP))
print('The min peak is ' + str(minP))

# Output:
#The max peak is -0.0578991
#The min peak is -0.129974
# I think these have units of m/s^2

# Well that looks silly.  We should remove the linear trend because there is an off-set
data = detrend(data)

# Okay find the min and max again
maxP = max(data)
minP = min(data)
print('The max peak is ' + str(maxP))
print('The min peak is ' + str(minP))

# Output:
#The max peak is 0.0364919
#The min peak is -0.0355832
# I think these have units of m/s^2

# So the peak to peak would be
print('P to P: ' + str(abs(minP) + abs(maxP)))
# Output: P to P: 0.0720751

# What about converting to m/s or m?
dataVelocity = cumtrapz(data,x=None, dx=0.01)
dataDisplacement = cumtrapz(dataVelocity,x=None, dx=0.01)
# Cool. That must be how you take an integral in Python.

# Check what cumtrapz returns.  Notice we need a different time vector.
# cumtrapz computes an approximation of the cumulative integral of 
#Y via the trapezoidal method with unit spacing
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
#decimate essentially means divide by, so if our sampling frequency is 100 and
#we decimate by 10, then our sampling frequency is 10
#When we decimate by 10, the Nyquist then becomes 5 Hz.
dataDec = decimate(data,10)
print('Here is dataDec len: ' + str(len(dataDec))) 
#Output: Here is dataDec len: 8557

# Why do we need a new time vector?
#Since we are taking 1/10 of the measurements, we need a new time vector that
#is also 1/10 of the original time vector.
t = np.arange(len(dataDec))/10.
fig = plt.figure(1)
plt.plot(t,dataDec)
plt.savefig('AccelExample6.jpg')
#plt.show()
plt.close()


## Ask Adam about this part - Is the idea to low-pass dataDec (as the acceleration), or the displacement or velocity?



# If I wanted to low-pass dataDec what would I need to change in my filter parameters?
# Answer - 
corner_dec = 0.1
b,a = butter(order, corner_dec/nyq, btype='high', analog=False)
dataDec = lfilter(b,a,dataDec)
fig = plt.figure(1)
plt.plot(t,dataDec)
plt.savefig('AccelExample6a.jpg')
#plt.show()
plt.close()


# Now we want to low-pass the data at 1 Hz
# Define some parameters so we know what is going on
order = 2
fs = 10.0
corner = 1.

# If my sampling rate is 10 Hz what is my nyquist?  Notice we decimated the data
nyq = 0.5 * fs
#Nyquist becomes 5

# Look up this function what kind of filter is this?
b, a = butter(order, corner/ nyq, btype='low', analog=False)

dataLP = lfilter(b,a,dataDec)
#lfilter is a low pass butterworth filter

fig = plt.figure(1)
plt.plot(t,dataLP)
plt.savefig('AccelExample7.jpg')
#plt.show()
plt.close()


#  Okay last part  Why don't we figure out what the PSD is?
# f is the frequency vector and P is the power
fs= 100.
f, P = welch(data, fs, nperseg = 512)
#welch's method computer an esimate of the power spectral density by dividing
#the data into overlapping segments.

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

## Start here.

# Okay one last exercise.  What happens when we compute the PSD of our velocity trace?
# If we calculate the velocity signal's power intensity in the frequency domain, my 
# initial thought is that we'd get a straight line, since the units will both be 
# "per second," but I know this isn't correct.  Right, that doesn't make sense.

fV, PV = welch(dataVelocity, 100., nperseg = 512)

# What are the units of PV?  I'm not sure about this either.
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
