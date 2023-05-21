import matplotlib.pyplot as plt
# Importing the required libraries
import numpy as np
from scipy import signal as sg
from scipy.signal import hilbert


# Function to generate an MSK signal given a bit sequence, sample rate, bit rate, and carrier frequency
def generate_msk_signal(bit_sequence, sample_rate, bit_rate, carrier_freq):
    # Generate an array of time samples based on the bit rate and sample rate
    time_samples = np.arange(0, len(bit_sequence) / bit_rate, 1 / sample_rate)
    # Initialize the MSK signal as a complex-valued array of zeros with the same shape as the time samples array
    msk_signal = np.zeros_like(time_samples, dtype=complex)

    # Iterate through the bit sequence and generate the MSK signal
    for idx, bit in enumerate(bit_sequence):
        # Define the start and end times for the current bit
        t_start = idx / bit_rate
        t_end = (idx + 1) / bit_rate
        # Extract the time samples for the current bit
        t_bit = time_samples[(time_samples >= t_start) & (time_samples < t_end)]

        # Calculate the phase and frequency for the current bit
        phase = np.pi * bit + np.pi / 2
        freq = carrier_freq + (-1)**bit * bit_rate / 4
        # Calculate the MSK signal for the current bit and update the overall MSK signal
        msk_signal[(time_samples >= t_start) & (time_samples < t_end)] = np.exp(1j * (2 * np.pi * freq * t_bit + phase))

    return msk_signal, time_samples

# Function to generate QAM constellation points for a given QAM order (m x m)
def qam_constellation_points(m):
    qam_points = np.zeros((m, m), dtype=complex)
    amplitude = np.sqrt(1 / (2 * (m - 1)))
    for i in range(m):
        for j in range(m):
            qam_points[i, j] = amplitude * (2 * i - m + 1) + 1j * amplitude * (2 * j - m + 1)
    return qam_points.ravel()

# Function to calculate the mean squared error between two signals (MSK and OFDM)
def objective_function(msk_signal, ofdm_signal):
    return np.mean(np.abs(msk_signal - ofdm_signal)**2)

# Function to find the best QAM constellation point to minimize the mean squared error between the MSK and OFDM signals
def best_qam_point(msk_signal, time_samples, frequency, qam_points):
    min_error = float('inf')
    best_point = None

    for point in qam_points:
        ofdm_signal = point * np.exp(1j * 2 * np.pi * frequency * time_samples)
        error = objective_function(msk_signal, ofdm_signal)
        if error < min_error:
            min_error = error
            best_point = point

    return best_point

# Define the parameters for the MSK signal
sample_rate = 5000000
bit_rate = 1000000
carrier_freq = 2400000000
msk_bandwidth = 2000000
bit_sequence = np.random.randint(0, 2, 100)

# Generate the MSK signal and time samples
msk_signal, time_samples = generate_msk_signal(bit_sequence, sample_rate, bit_rate, carrier_freq)

# Define the number of subcarriers and their frequencies
num_subcarriers = 48
frequencies = np.linspace(carrier_freq - msk_bandwidth / 2, carrier_freq + msk_bandwidth / 2, num_subcarriers)

# Generate the QAM constellation points for 1024 QAM (32 x 32)
qam_points = qam_constellation_points(32)

# Initialize the array of complex-valued amplitudes and phase shifts for each subcarrier
amplitudes_and_phase_shifts = np.zeros(num_subcarriers, dtype=complex)

# Iterate through the subcarriers and find the best QAM constellation point for each subcarrier
for k in range(num_subcarriers):
    amplitudes_and_phase_shifts[k] = best_qam_point(msk_signal, time_samples, frequencies[k], qam_points)

print("Optimized amplitudes and phase shifts:", amplitudes_and_phase_shifts)


# Function to generate the OFDM signal given amplitudes and phase shifts, time samples, and frequencies
def generate_ofdm_signal(amplitudes_and_phase_shifts, time_samples, frequencies):
    # Initialize the OFDM signal as a complex-valued array of zeros with the same shape as the time samples array
    ofdm_signal = np.zeros_like(time_samples, dtype=complex)
    # Iterate through the subcarriers and update the OFDM signal based on the amplitudes and phase shifts
    for k in range(len(frequencies)):
        ofdm_signal += amplitudes_and_phase_shifts[k] * np.exp(1j * 2 * np.pi * frequencies[k] * time_samples)
    return ofdm_signal

# Generate the OFDM signal using the optimized amplitudes and phase shifts
ofdm_signal = generate_ofdm_signal(amplitudes_and_phase_shifts, time_samples, frequencies)

# Calculate the mean squared error between the MSK signal and the generated OFDM signal
mse = objective_function(msk_signal, ofdm_signal)

# Compare the mean squared error with a threshold value
threshold = 1e-6
if mse < threshold:
    print("The output values are valid. Mean Squared Error: ", mse)
else:
    print("The output values may not be valid. Mean Squared Error: ", mse)

# Plotting the signals
plt.figure(figsize=(15, 10))

# MSK Signal
plt.subplot(3, 1, 1)
plt.plot(time_samples[:1000], np.real(msk_signal[:1000]), label='MSK signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('MSK Signal')
plt.grid(True)
plt.legend()

# Original OFDM Signal
plt.subplot(3, 1, 2)
ofdm_signal_original = np.zeros_like(time_samples, dtype=complex)
for k in range(len(frequencies)):
    ofdm_signal_original += np.exp(1j * 2 * np.pi * frequencies[k] * time_samples)
plt.plot(time_samples[:1000], np.real(ofdm_signal_original[:1000]), label='Original OFDM signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original OFDM Signal')
plt.grid(True)
plt.legend()

# Modified OFDM Signal
plt.subplot(3, 1, 3)
plt.plot(time_samples[:1000], np.real(ofdm_signal[:1000]), label='Modified OFDM signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Modified OFDM Signal')
plt.grid(True)
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()


# Define the demodulate_msk function
def demodulate_msk(signal, sample_rate, bit_rate):
    # Hilbert Transform to get the analytic signal
    analytic_signal = hilbert(signal)
    print('Analytic signal calculated')

    # Low Pass Filter to remove high frequency noise
    nyq_rate = sample_rate / 2.0
    width = 0.1/nyq_rate
    ripple_db = 60.0
    N, beta = sg.kaiserord(ripple_db, width)  # Here, use sg instead of signal
    cutoff_hz = bit_rate / 2.0
    taps = sg.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))  # Here, use sg instead of signal
    filtered_signal = sg.lfilter(taps, 1.0, np.real(analytic_signal))  # Here, use sg instead of signal
    print('Signal filtered with Low Pass Filter')

    # Differentiate Phase to get the original bits
    phase = np.unwrap(np.angle(filtered_signal))
    differentiated_phase = np.diff(phase)
    print('Phase of the signal differentiated')

    # Resample at bit intervals
    resampled_phase = differentiated_phase[::int(sample_rate/bit_rate)]
    print('Phase-differentiated signal resampled')

    # Decode the bits
    decoded_bits = np.array(resampled_phase > 0, dtype=int)
    print('Bits decoded')

    return decoded_bits


msk_demodulated_bits = demodulate_msk(np.real(msk_signal), sample_rate, bit_rate)
ofdm_demodulated_bits = demodulate_msk(np.real(ofdm_signal), sample_rate, bit_rate)

bit_errors = np.sum(np.abs(msk_demodulated_bits - ofdm_demodulated_bits))
bit_error_rate = bit_errors / len(bit_sequence)

print("Bit Error Rate:", bit_error_rate)
