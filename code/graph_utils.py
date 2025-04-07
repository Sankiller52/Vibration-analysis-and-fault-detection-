import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import pandas as pd

def plot_fft_with_anomalies(data, fault_results):
    st.subheader("🔍 Frequency Spectrum (FFT)")
    show_all = st.toggle("Show All FFT Graphs", value=True)
    current_processed = {
    "time_domain": pd.DataFrame(...),
    "fft": {
        "sensor1": {
            "frequencies": [...],
            "magnitudes": [...],
        }
        
    }
    
}


    for col,fft_info in data["fft"].items():
        if not show_all and col != list(data["fft"].keys())[0]:
            continue

        freq = fft_info["freq"]
        magnitude = fft_info["magnitude"]
        is_anomaly = col in fault_results.get("anomalies", [])

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(freq, magnitude, color='blue')
        ax.set_title(f"FFT of {col}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

        if is_anomaly:
            ax.set_facecolor("#ffe6e6")
            ax.annotate("Anomaly Detected", xy=(freq[np.argmax(magnitude)], max(magnitude)),
                        xytext=(20, max(magnitude)*0.8),
                        arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10,
                        color="red")

        st.pyplot(fig)


def plot_statistics_summary(stats):
    st.subheader("📈 Statistical Summary")

    for stat_name, values in stats.items():
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.bar(values.keys(), values.values(), color='skyblue')
        ax.set_title(stat_name.replace("_", " ").title())
        ax.set_ylabel("Value")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values.keys(), rotation=45)
        st.pyplot(fig)


def plot_time_domain_signals(df, title="📉 Time-Domain Signals"):
    st.subheader(title)
    for col in df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df[col], color='purple')
        ax.set_title(f"{col} - Time Domain")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        st.pyplot(fig)


def plot_overlay_comparison(no_fault_df, current_df):
    st.subheader("🔁 Overlay Comparison with No-Fault Data")
    for col in current_df.columns:
        if col not in no_fault_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(no_fault_df[col], label="No Fault", color='green', alpha=0.7)
        ax.plot(current_df[col], label="Current", color='red', alpha=0.7)
        ax.set_title(f"{col} - Overlay")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


def plot_spectrogram(df):
    st.subheader("🎛 Spectrogram / STFT View")
    for col in df.columns:
        f, t, Sxx = spectrogram(df[col].values, fs=1000)  # assuming fs=1000Hz
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_title(f"Spectrogram of {col}")
        st.pyplot(fig)
