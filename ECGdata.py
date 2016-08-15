import pylab
import scipy.signal as signal
import numpy

# The "Daubechies" wavelet is a rough approximation to a real,
# single, heart beat ("pqrst") signal            
pqrst = signal.wavelets.daub(10)

# Add the gap after the pqrst when the heart is resting.
samples_rest = 10
zero_array = numpy.zeros(samples_rest, dtype=float)
pqrst_full = numpy.concatenate([pqrst,zero_array])

# Simulated Beats per minute rate
# For a health, athletic, person, 60 is resting, 180 is intensive exercising
bpm = 60
bps = bpm / 60

# Simumated period of time in seconds that the ecg is captured in     
capture_length = 100

# Caculate the number of beats in capture time period
# Round the number to simplify things             
num_heart_beats = int(capture_length * bps)

# Concatonate together the number of heart beats needed
ecg_template = numpy.tile(pqrst_full , num_heart_beats)

# Add random (gaussian distributed) noise
noise = numpy.random.normal(0, 0.01, len(ecg_template))
ecg_template_noisy = noise + ecg_template


#print(ecg_template_noisy.shape)
#print(type(ecg_template_noisy))

# Plot the noisy heart ECG template
pylab.plot(ecg_template_noisy)
pylab.xlabel('Sample number')
pylab.ylabel('Amplitude (normalised)')
pylab.title('Heart ECG Template')

