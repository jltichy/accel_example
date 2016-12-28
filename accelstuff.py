#!/usr/bin/env python

import numpy as np # This is the package needed for functions.
from scipy.signal import butter, lfilter, freqz, detrend, decimate, welch
from scipy.integrate import cumtrapz
import math
import sys
import matplotlib
matplotlib.use('Agg') #This is required to run matplotlib on Google Chrome.
import matplotlib.pyplot as plt

# Here is some accelerometer data
f = open('acceldata','r') # The 'r' means read the file.

# Read in the data as a list.
data = []   # data is an empty list
for line in f: # for each line in data, append a string of information.
    data.append(int(line)) # The data came in as a string and we will need to
    # convert it into integers.  However, as we'll see below, we may not
    # necessarily only have integers when we convert our counts to acceleration.
    # Therefore, we will eventually need floats (data types that can have
    # decimals).

# The data is sampled at 100 Hz.  What is the time length of the file?
# One Hz is 1/sec, so that means we are taking 100 samples every second.
print('Here is the time span: ' + str(float(len(data))/100.) + ' seconds')
# Here is the time span: 855.7 seconds

# Let's print the number of samples:
print('There are ' + str(float(len(data))) + ' samples.')
# Result - There are 85570.0 samples.

# Accelerometers output a voltage which is proportional to acceleration.
# The acceleration is then digitized into counts.  This is done through a
# device called a digitizer

# If the digitizer is a 24 bit digitizer with a 40 Volt peak to peak range,
# then the sensitivity is 2^24/40 which is counts/V.
# Your data is in counts. Convert it to Volts.

data = np.asarray(data,dtype=np.float32)/((2**24)/40.)
# Another way to do the conversion is like this:
# data = np.asarray(data,dtype=np.float32)*40/(2**24) because 2**24 counts = 40V
# and we are converting from counts to volts.  Again it's these voltages that
# are proportional to acceleration.
# float32 is the type of data (it's 32 bit data), meaning that one bit assigns
# the sign, 8 bits assign the exponent, and 23 bits are for the mantissa (sig figs)
# 32 bits is all the precision we need.

print(data) # Notice that these are float32 data types, with decimal places

# Here is the data in volts: 
#[-0.09621859 -0.0962162  -0.09622097 ..., -0.0949502  -0.09496927
# -0.09504557]

# Now that your data is in V you want to convert it to m/s^2 to get acceleration.
# The sensitivity of your accelerometer is 2g = 20 V.  g is gravity, so we
# multiply by 2 and gravity and divide by 20 to convert to acceleration.

data = data *2.*9.81/20.

print data

# Here is the data in m/s^2 (acceleration):
#[-0.09439044 -0.0943881  -0.09439278 ..., -0.09314615 -0.09316486
# -0.09323971]

# Now plot your data to see what you have.
# Grab a time vector (t).

# Why are we dividing by 100? Think sample rate.  The data is sampled at 100Hz.
t=np.arange(len(data))/100.

fig = plt.figure()
plt.plot(t,data)
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration vs. Time')
plt.savefig('AccelExample1.jpg')
plt.close()

# Now we want to low-pass the data at 10 Hz.

# Define some parameters so we know what is going on.

order = 2 # Order is the order of the filter - What does this mean??
# A second order filter decreases at -12 dB per octave (for every factor of 2
# in the frequency, we lose 12 dB).  The higher the order, the steeper the data
# will drop off.

fs = 100.0 # fs is the sampling rate, in Hz

corner = 10. # corner is the frequency cut off, so we will keep all data less
# than 10 Hz and everything greater than 10 Hz will be attenuated.  Remember that
# the higher the order, the steeper the data will drop off, at this 10 Hz corner.

# If my sampling rate is 100 Hz, what is my nyquist? Nyquist is half the
# sampling rate, so it's 50 Hz in this case
nyq = 0.5 * fs

# Look up this function. What kind of filter is this?

# Butterworth Filter - a signal processing filter that aims for  
# as flat a frequency response as possible in the passband (aka band pass). 
b, a = butter(order, corner/nyq, btype='low', analog=False)

# b is the numerator and a is the denominator of the polynomials in the filter.
# we do corner/nyq as 10/50 because these are the critical frequencies.
# btype = 'low' tells the filter to do low pass 
# analog = False means that we want the digital filter, not an analog filter.
# Digital filters perform better than analog.

# Let's calculate the low pass data.  lfilter is the infinite impulse response
# filter (most efficient type) that takes in b, the numerator coefficient
# vector in the 1-D sequence, and a, the denominator coefficient of the 1-D 
# sequence.  data is our accelerometer data in m/s^2
dataLP = lfilter(b,a,data)

# Now plot both the low-pass and the regular data

fig = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,data)
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration vs. Time')
plt.subplot(2,1,2)
plt.plot(t,dataLP)
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Low Pass Acceleration at Corner of 10 Hz. vs. Time')
plt.savefig('AccelExample2.jpg')
plt.close()

# It looks like the difference between these two plots is that the dataLP
# plot looks squished, but it seems like that data is the same...Yes, that is
# the point of a low pass filter.  The lower frequencies came through and the
# higher ones got attenuated.

# Now change the corner to make this a 0.1 Hz low pass filter.
corner2 = 0.1
b, a = butter(order, corner2/nyq, btype='low', analog=False)
dataLP = lfilter(b,a,data)
fig = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,data)
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration vs. Time')
plt.subplot(2,1,2)
plt.plot(t,dataLP)
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Low Pass Acceleration at Corner of 0.1 Hz. vs. Time')
plt.savefig('AccelExample2a.jpg')
plt.close()
# Even more high frequency data has been attenuated and now we are looking at
# only very low frequencies.  The data drops off at the 0.1 Hz corner in this 
# case.  Before, the corner was set to 10 Hz.  Again, the order of the filter
# determines how steeply the data drops off at the defined corner.  The higher
# the order of the filter, the steeper it drops off.

# The spike at the front is called filter ringing and is annoying.
# We can get rid of that by applying a taper.
taper = np.hanning(len(data))
# The Hanning window is used to select a subset of samples.  It minimizes 
# aliasing (distortion).  In this case, the distortion is called filter ringing.

dataLP = lfilter(b,a,data*taper)

fig = plt.figure(1)
plt.plot(t,dataLP)
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Low Pass Acceleration at Corner of 0.1 Hz. with Hanning Window')
plt.savefig('AccelExample3.jpg')
plt.close()

# Well now the data is even more ugly, but we can fix this by applying a high
# pass filter.  Now everything below 0.1 Hz will go through, as well as every-
# thing above 0.01 Hz.  The range is now between 0.01 and 0.1 Hz.
cornerHP = 0.01
b,a = butter(order, cornerHP/nyq, btype='high', analog=False)
dataLP = lfilter(b,a,dataLP)
fig = plt.figure(1)
plt.plot(t,dataLP)
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Band Pass Acceleration between 0.1 Hz and 10 Hz with Hanning Window')
plt.savefig('AccelExample4.jpg')
plt.close()

# What is the frequency content of dataLP?  
# Hint: We let frequencies lower than 10 Hz make it through.
# We also let frequencies of greater than 0.01 Hz make it through.

# The range is now between 0.01 and 0.1 Hz.  This is the band-pass!



####  Okay we should change focus. Let's go back to the original data.

# What is the peak to peak acceleration? Note that we are going back to the
# original data.
maxP = max(data)
minP = min(data)
print('The max peak is ' + str(maxP))
print('The min peak is ' + str(minP))

# Output:
#The max peak is -0.0578991
#The min peak is -0.129974
# I think these have units of m/s^2

# Well that looks silly (both the min and max peaks are negative).  
# We should remove the linear trend because there is an off-set.
data = detrend(data)
# detrend removes the mean value or linear trend from a vector or matrix

# Okay find the min and max again.
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

# What about converting to m/s or m?  The function cumtrapz calculates the 
# area under the curve.  The area under the acceleration curve, using the 
# trapezoidal rule, is velocity, and the area under the velocity curve, using 
# the trapezoidal rule, is displacment.
dataVelocity = cumtrapz(data,x=None, dx=0.01)
dataDisplacement = cumtrapz(dataVelocity,x=None, dx=0.01)
# Cool. That must be how you take an integral in Python.
# x=None means that the function should use spacing dx between consecutive
# elements of "data".  dx is then defined to be 0.01.

# Check what cumtrapz returns.  Notice we need a different time vector.
# cumtrapz computes an approximation of the cumulative integral of 
#Y via the trapezoidal method with unit spacing
# Since cumtrapz uses spacing of 0.01, we'll need to change the time vectors.

tv = np.arange(len(dataVelocity))/100.
td = np.arange(len(dataDisplacement))/100.

fig = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(tv,dataVelocity)
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity and Displacement using cumptraz Integration')
plt.subplot(2,1,2)
plt.plot(td,dataDisplacement)
plt.xlabel('Time (seconds)')
plt.ylabel('Displacement (m)')
plt.savefig('AccelExample5.jpg')
plt.close()

# Note most accelerometers don't integrate to displacement well because of noise.
# That is the trail that you see in "AccelExample5"

# What happens to the length and the Nyquist when we decimate by a factor of 10?
#decimate essentially means divide by, so if our sampling frequency is 100 and
#we decimate by 10, then our sampling frequency is 10
#When we decimate by 10, the Nyquist then becomes 5 Hz.

# Side note - there are multiple ways to decimate.  Taking every 10th data point,
# as we have done here, is just one method.

dataDec = decimate(data,10)
print('Here is dataDec len: ' + str(len(dataDec))) 
#Output: Here is dataDec len: 8557

# Why do we need a new time vector?
#Since we are taking 1/10 of the measurements, we need a new time vector that
#is also 1/10 of the original time vector.
t = np.arange(len(dataDec))/10.
fig = plt.figure(1)
plt.plot(t,dataDec)
plt.xlabel('Time (seconds)')
plt.ylabel('Decimated Acceleration (m/s^2)')
plt.title('Decimated Acceleration')
plt.savefig('AccelExample6.jpg')
plt.close()

# If I wanted to low-pass dataDec what would I need to change in my filter parameters?
# Answer - 
corner_dec = 0.1
b,a = butter(order, corner_dec/nyq, btype='high', analog=False)
dataDec = lfilter(b,a,dataDec)
fig = plt.figure(1)
plt.plot(t,dataDec)
plt.title('Decimated Acceleration with Low Pass Corner of 0.1 Hz')
plt.xlabel('Time (seconds)')
plt.ylabel('Decimated Acceleration (m/s^2)')
plt.savefig('AccelExample6a.jpg')
plt.close()

# Let's create a figure that shows the difference between the decimated data
# and the low pass decimated data.
fig = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t,dataDec)
plt.xlabel('Time (seconds)')
plt.ylabel('Decimated Acceleration (m/s^2)')
plt.title('Decimated Accel and Decimated Accel with Low Pass Corner of 0.1 Hz')
plt.subplot(2,1,2)
plt.plot(t,dataDec)
plt.xlabel('Time (seconds)')
plt.ylabel('Decimated Acceleration (m/s^2)')
plt.savefig('AccelExample6b.jpg')
plt.close()

# Now we want to low-pass the data at 1 Hz.
# Define some parameters so we know what is going on.
order = 2
fs = 10.0
corner = 1.

# If my sampling rate is 10 Hz, what is my nyquist?  Notice we decimated the data.
nyq = 0.5 * fs
#Nyquist becomes 5

b, a = butter(order, corner/nyq, btype='low', analog=False)

dataLP = lfilter(b,a,dataDec)
#lfilter is a low pass butterworth filter - see description above

fig = plt.figure(1)
plt.plot(t,dataLP)
plt.xlabel('Time (seconds)')
plt.ylabel('Decimated Acceleration (m/s^2)')
plt.title('Decimated Acceleration with Low Pass Corner of 1 Hz')
plt.savefig('AccelExample7.jpg')
plt.close()


# Okay last part  
# Why don't we figure out what the PSD is?
# f is the frequency vector and P is the power
fs= 100. # Again, this is the sampling frequency
f, P = welch(data, fs, nperseg = 512)
# Welch's method - compute an esimate of the power spectral density by dividing
# the data into overlapping segments.
# nperseg is the length of each segment.  The default is 256 but we set it to
# 512 here.

# The units of P are (m/s^2)^2 /Hz, which is confusing
fig = plt.figure(1)
plt.plot(f,P)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Accelerometer Power (m/s^2)^2 /Hz')
plt.title('PSD of Original Accelerometer Data')
plt.savefig('AccelExample8.jpg')
plt.close()

# These units are messy, so if we use Hz=1/sec, we can simplify the units to
# m^2/s^3 but maybe this doesn't really make sense either.

# Yuck. That is hard to read. Why not plot it on a log scale?
fig = plt.figure(1)
plt.semilogx(f,P)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Accelerometer Power (m/s^2)^2 /Hz')
plt.title('Semi-Log PSD of Original Accelerometer Data')
plt.savefig('AccelExample9.jpg')
plt.close()

# Better, but we could convert it to dB with the following eqn: 10*log10(P)?
PdB = 10.*np.log10(P) # This will be the power in decibels.
fig = plt.figure(1)
plt.semilogx(f,PdB)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Accelerometer Power (dB)')
plt.title('Semi-Log PSD of Original Accelerometer Data, displayed in Decibels')
plt.savefig('AccelExample10.jpg')
plt.close()

# Okay, what if we compute the power for our Low-pass data?
fLP, PLP = welch(dataLP,10., nperseg = 512)

fig = plt.figure(1)
plt.semilogx(f,PdB, label='Original Data')
plt.semilogx(fLP, 10*np.log10(PLP), label='Low-Pass Data') #convert to decibels here.
plt.legend(loc='upper right')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Low Pass Accelerometer Power (dB)')
plt.title('Semi-Log PSD of Original and Low-Pass Accelerometer Data, displayed in Decibels')
plt.savefig('AccelExample11.jpg')
plt.close()

# Which is PLP and which is PdB?  Hint: One of them should not have as much 
# power past 10Hz.
# Answer - The green line is the low-pass data (PLP) and the blue line is the 
# original data (PdB).  It's beneficial to see the two lines on one plot.

# Why do they have different frequencies?
# PLP is the PSD of the low pass data
# PdB is the PSD in decibels of the original data

# Okay one last exercise.  What happens when we compute the PSD of our velocity
# trace?
# You get units of (m/s)^2/Hz if you take the PSD of velocity data.

fV, PV = welch(dataVelocity, 100., nperseg = 512)

# What are the units of PV? (m/s)^2/Hz

# Side note - the units for the PSD of acceration are (m/s^2)^2/Hz

# For an exercise, let's see what happens if we were to take the PSD of 
# displacement data.  I think it would be m^2*s.  Ask Adam if this is right.
# Yes, this is right.  Another way to display the units is m^2/Hz.  I think
# it is better to use the Hz version to be consistent across displacement,
# velocity, and acceleration.

# Units: 
# http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html

fig = plt.figure(1)
plt.semilogx(fV, 10*np.log10(PV), label='Velocity Trace') #Here we convert to decibels again.
plt.semilogx(f,PdB, label='Acceleration Trace')
plt.legend(loc='upper right')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.title('Semi-Log PSD of Velocity and Acceleration, displayed in Decibels')
plt.savefig('AccelExample12.jpg')
plt.close()

# What happens if we multiply PV by omega^2=(2*pi*f)^2?
PA = 10.*np.log10(PV*(2*np.pi*fV)**2)
# We get angular frequency.  This gives the normalized frequency for PSD.
# It's a convention thing.  Some groups use angular fequency.  It's best to
# just be consistent, no matter which convention you choose.
# The values are a little more familiar and the data is easier to understand.
# It's a convention to see how fast something is spinning.
# units = (2pi*m/s)^2

# What are the units of PA?  Why are these so similar?  
# accelerometer power units = (m/s^2)^2 /Hz
# velocity power units = (m/s)^2/Hz
# displacement power units = m^2/Hz

fig = plt.figure(1)
plt.semilogx(fV,PA, label='Velocity PSD * omega')
plt.semilogx(f,PdB, label='Acceleration Trace PSD')
plt.legend(loc='upper right')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Semi-Log PSD, comparing Acceleration PSD to Velocity PSD times omega')
plt.savefig('AccelExample13.jpg')
plt.close()

# So to recap:
# You took the power spectra of the velocity.  You multiplyed it by omega and 
# got the same thing as computing the power spectra of the acceleration.

# But it's not exactly the same.
# That's where I'm confused.

# The "exactly the same" comes from precision and cumtrapz issues.
# When you integrate a spectra you introduce noise.
# So it is fairly close, but up to some differences in the approach.

# Ahh.  Got it.  Thanks.

# Take a look at 106 in the functional relationships
# https://en.wikipedia.org/wiki/Fourier_transform

# Why do we do (2*pi*f)^2 and not just (2*pi*f)?  Look at what Welch outputs.
# This is a bit different from the FFFFT

# We square the multiplier because it is acceleration, not velocity.

#End Note:
# The units of angular frequency for acceleration data are (2*pi*m)^2/s^4
# while the units of angular frequency for velocity data are (2*pi*m/s)^2
