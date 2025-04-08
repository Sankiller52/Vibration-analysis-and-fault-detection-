import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
import pandas as pd

def plot_fft_with_anomalies(df,fault_sensors,sensor_columns):
    st.subheader("🔍 Frequency Spectrum (FFT)")
    show_all = st.toggle("Show All FFT Graphs", value=True)

    for idx, sensor in enumerate(sensor_columns):
        if not show_all and idx > 0:
            break  # Show only first if toggle is off

        freq_col = f"{sensor}_fftfreq"
        amp_col = f"{sensor}_fftspectrum"

        if freq_col not in df.columns or amp_col not in df.columns:
            st.warning(f"Missing columns for {sensor}")
            continue

        freq = df[freq_col].values
        magnitude = df[amp_col].values
        is_anomaly = sensor in fault_sensors

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(freq, magnitude, color='blue')
        ax.set_title(f"FFT of {sensor}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

        if is_anomaly:
            ax.set_facecolor("#ffe6e6")
            ax.annotate("Anomaly Detected", 
                        xy=(freq[magnitude.argmax()], magnitude.max()), 
                        xytext=(freq[magnitude.argmax()]*0.6, magnitude.max()*0.8),
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        fontsize=10, color="red")

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

    common_cols = [col for col in current_df.columns if col in no_fault_df.columns][:2]  # Try 2 for now

    if not common_cols:
        st.warning("⚠️ No common sensor columns found.")
        return

    for col in common_cols:
        try:
            x1 = no_fault_df[col].dropna().values
            x2 = current_df[col].dropna().values
            min_len = min(len(x1), len(x2))

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(x1[:min_len], label="No Fault", color='green', alpha=0.7)
            ax.plot(x2[:min_len], label="Current", color='red', alpha=0.7)
            ax.set_title(f"{col} - Overlay")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Overlay failed for {col}: {e}")

    st.success("✅ Overlay comparison done!")



def plot_spectrogram(df,cl):
    st.subheader("🎛 Spectrogram / STFT View")
    for col in cl:
        f, t, Sxx = spectrogram(df[col].values, fs=1000)  # assuming fs=1000Hz
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_title(f"Spectrogram of {col}")
        st.pyplot(fig)
