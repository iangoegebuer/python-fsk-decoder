import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
import getopt,sys

dataFile = ''
outputfile = ''
# offset frequency in Hz (read from the previous plot)
offset_frequency = 0

# limit the sampling rate using decimation, let's use the decimation by 4
bb_dec_factor = 300

# bitrate assumption, will be corrected for using early-late symbol sync (as
# read from the spectral content plot from above)
bit_rate = 1250

helpstring = '''
Decode.py -i <data file> -o <demodulated file> -d <decimation amount> -b <bitrate> -f <offset frequency>
'''

try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:d:f:",[])
except getopt.GetoptError:
    print(helpstring)
    exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(helpstring)
        exit()
    elif opt in ("-i"):
        dataFile = arg
    elif opt in ("-o"):
        outputfile = arg
    elif opt in ("-d"):
        bb_dec_factor = int(arg)
    elif opt in ("-b"):
        bit_rate = int(arg)
    elif opt in ("-f"):
        offset_frequency = float(arg)

#
# DATA LOAD
#

# read the wave file
fs, rf = wf.read(dataFile)
# get the scale factor according to the data type
sf = {
    np.dtype('int16'): 2**15,
    np.dtype('int32'): 2**32,
    np.dtype('float32'): 1,
}[rf.dtype]

# convert to complex number c = in_phase + j*quadrature and scale so that we are
# in (-1 , 1) range
rf = (rf[:, 0] + 1j * rf[:, 1]) / sf

# plot the rf signal to see the transmitting station offset
plt.subplot(3, 3, 1)
plt.title('RF spectrum')
plt.psd(rf, Fs=fs)


#
# MIX TO BASEBAND
#

# baseband local oscillator
bb_lo = np.exp(1j * (2 * np.pi * (-offset_frequency / fs) *
                      np.arange(0, len(rf))))

# complex-mix to bring the rf signal to baseband (so that is centered around
# something around 0Hz. doesn't have to be perfect.
bb = rf * bb_lo

# plot the mixed signal which ought to bring the signal of interest to the
# baseband
plt.subplot(3, 3, 2)
plt.title('Baseband Spectrum')
plt.psd(bb, Fs=fs)


#
# DECIMATE
#

# get the baseband sampling frequency
bb_fs = fs // bb_dec_factor
# let's prepare the low pass decimation filter that will have a cutoff at the
# half of the bandwidth after the decimation
dec_lp_filter = sig.butter(3, 1 / (bb_dec_factor * 2))
# filter the signal
bb = sig.filtfilt(*dec_lp_filter, bb)
# decimate
bb = bb[::bb_dec_factor]

# plot the result of the decimation
plt.subplot(3, 3, 3)
plt.title('Decimated Baseband Spectrum')
plt.psd(bb, Fs=bb_fs)


#
# SELECT THE PART OF THE DATA THAT CONTAINS THE TRANSMISSION
#

# plot the whole signal
plt.subplot(3, 3, 4)
plt.title('Whole Signal magnitude, time domain')
plt.xlabel('Time [s]')
plt.ylabel('Magnitude')
plt.plot(np.arange(0, len(bb)) / bb_fs, np.abs(bb))

# using the signal magnitude let's determine when the actual transmission took
# place
bb_mag = np.abs(bb)
# mag threshold level (as read from the chart above)
bb_mag_thrs = 0.02

# indices with magnitude higher than threshold
bb_indices = np.nonzero(bb_mag > bb_mag_thrs)[0]
# limit the signal
bb = bb[np.min(bb_indices) : np.max(bb_indices)]

# plot the selected signal
plt.subplot(3, 3, 5)
plt.title('Selected Signal')
plt.xlabel('Time [s]')
plt.ylabel('Magnitude')
plt.plot(np.arange(0, len(bb)) / bb_fs, np.abs(bb))

#
# DEMODULATION
#

# demodulate the fm transmission using the difference between two complex number
# arguments. multiplying the consecutive complex numbers with their respective
# conjugate gives a number who's angle is the angle difference of the numbers
# being multiplied
bb_angle_diff = np.angle(bb[:-1] * np.conj(bb[1:]))
# mean output will tell us about the frequency offset in radians per sample
# time. If the mean is not zero that means that we have some offset. lets' get
# rid of it
dem = bb_angle_diff - np.mean(bb_angle_diff)

# plot the demodulated signal spectrum in order to get a ballpark estimate about
# the bitrate: it will display itself as a null in spectrum since random binary
# streams have spectral nulls in n * 1/Tsymbol
plt.subplot(3, 3, 6)
plt.title('Demodulated Signal Spectrum,')
plt.psd(dem, Fs=bb_fs)

# show the demodulated data in time domain
plt.subplot(3, 3, 7)
plt.title('Demodulated Signal,')
plt.xlabel('Time [s]')
plt.ylabel('Value')
plt.plot(np.arange(0, len(dem)) / bb_fs, dem)


if outputfile != '':
    wf.write(outputfile, bb_fs, dem)

#
# SIGNAL SYNCHRONIZATION (DATA RECOVERY)
#

# time to sample symbols, let's use the early-late (el) symbol synchronization
# scheme with the numerically controlled oscillator (nco) for the actual
# sampling

# calculate the nco step based on the initial guess for the bit rate. Early-Late
# requires sampling 3 times per symbol
nco_step_initial = bit_rate * 3 / bb_fs
# use the initial guess
nco_step = nco_step_initial
# phase accumulator values
nco_phase_acc = 0
# samples queue
el_sample_queue = []

# couple of control values
nco_steps, el_errors, el_samples = [], [], []

# process all samples
for i in range(len(dem)):
    # current early-late error
    el_error = 0
    # time to sample?
    if nco_phase_acc >= 1:
        # wrap around
        nco_phase_acc -= 1

        # alpha tells us how far the current sample is from perfect
        # sampling time: 0 means that dem[i] matches timing perfectly, 0.5 means
        # that the real sampling time was between dem[i] and dem[i-1], and so on
        alpha = nco_phase_acc / nco_step
        # linear approximation between two samples
        sample_value = (alpha * dem[i - 1] + (1 - alpha) * dem[i])
        # append the sample value
        el_sample_queue += [sample_value]

        # got all three samples?
        if len(el_sample_queue) == 3:
            # get the early-late error: if this is negative we need to delay the
            # clock
            if el_sample_queue:
                el_error = (el_sample_queue[2] - el_sample_queue[0]) / \
                           -el_sample_queue[1]
            # clamp
            el_error = np.clip(el_error, -10, 10)
            # clear the queue
            el_sample_queue = []
        # store the sample
        elif len(el_sample_queue) == 2:
            el_samples += [(i - alpha, sample_value)]

    # integral term
    nco_step += el_error * 0.01
    # sanity limits: do not allow for bitrates outside the 30% tolerance
    nco_step = np.clip(nco_step, nco_step_initial * 0.7, nco_step_initial * 1.3)
    # proportional term
    nco_phase_acc += nco_step + el_error * 0.3

    # append
    nco_steps += [nco_step]
    el_errors += [el_error]


# show the sampling points
plt.subplot(3, 3, 8)
plt.title('Sampling Points,')
plt.xlabel('Time [s]')
plt.ylabel('Value')
plt.plot(np.arange(0, len(dem)) / bb_fs, dem)
plt.plot([x[0] / bb_fs for x in el_samples], [x[1] for x in el_samples], ".")


#
# GRANDE FINALE
#

# display sampled data
plt.subplot(3, 3, 9)
plt.title('Data,')
plt.xlabel('Bit Number')
plt.ylabel('Value')

bits = (np.array([x[1] for x in el_samples]) >= .00 *1).astype(int)

last = 0
data = []
found_start = False
found_end = False
for x in bits:
    if found_start and found_end:
        data.append(x)
    last = last << 1
    last = last + x
    last = last & 0xff
    if(last == 0b10101010) and not found_start:
        found_start = True
        print("found start!")
    if found_start and (last != 0b10101010 and last != 0b1010101):
        found_end = True
bits = np.array(data)
bits = (bits[1:] - bits[0:-1]) % 2
bits = bits.astype(np.uint8)

plt.plot(bits)
# plt.plot([x[1] >= 0 for x in el_samples])

# finally, let's output the bit stream
print(bits)
print(len(bits))

# show all plots
plt.show()
