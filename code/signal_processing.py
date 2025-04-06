import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
from scipy.stats import kurtosis

# --- Butterworth Filter ---
def butterworth_filter(data, cutoff=50, fs=1000, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

# --- Frequency Spectrum (FFT) ---
def compute_fft(signal, sampling_rate=1000):
    n = len(signal)
    freq = np.fft.fftfreq(n, d=1 / sampling_rate)
    spectrum = np.abs(fft(signal))
    return freq[:n // 2], spectrum[:n // 2]

# --- Statistical Features ---
def calculate_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    p2p = np.ptp(signal)
    crest = np.max(np.abs(signal)) / rms if rms != 0 else 0
    kurt = kurtosis(signal)
    return {
        'RMS': rms,
        'PeakToPeak': p2p,
        'CrestFactor': crest,
        'Kurtosis': kurt
    }

# --- Main Processing Function ---
def process_signals(l, sensor_columns, sampling_rate=1000):
    df=pd.DataFrame()
    
    results = {}
    
    for sensor in sensor_columns:
        clean_signal = butterworth_filter(l[sensor].values, fs=sampling_rate)
        freq, spectrum = compute_fft(clean_signal)
        stats = calculate_features(clean_signal)

        # Save to results
        results[sensor] = {
            'filtered': clean_signal,
            'fft_freq': freq,
            'fft_amp': spectrum,
            'features': stats
        }

        # Append to DataFrame
        df[f'{sensor}_filtered'] = clean_signal
        df[f'{sensor}_rms'] = stats['RMS']
        df[f'{sensor}_p2p'] = stats['PeakToPeak']
        df[f'{sensor}_crest'] = stats['CrestFactor']
        df[f'{sensor}_kurtosis'] = stats['Kurtosis']

    return df
