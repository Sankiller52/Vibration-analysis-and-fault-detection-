import streamlit as st
import plotly.graph_objs as go
import numpy as np
import time
from scipy.signal import spectrogram

# Live Data Plot (continuously updating, pinned to top)
def live_data_plot(df, sensor_columns):
    st.markdown("## 📡 Live Monitoring (Raw Vibration Data)")
    
    # Create a container for each sensor's live graph
    for sensor in sensor_columns:
        graph_container = st.empty()

        # Simulate live feed (example: update last 200 points)
        for _ in range(1):  # Single update per run
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df[sensor].values[-200:],  # Show last 200 points
                mode='lines',
                name=sensor,
                line=dict(color='royalblue')
            ))
            fig.update_layout(
                title=f"Live - {sensor}",
                xaxis_title='Sample Index',
                yaxis_title='Amplitude',
                height=250,
                template='plotly_white',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            graph_container.plotly_chart(fig, use_container_width=True)
            time.sleep(0.5)  # To simulate update (adjust as needed)

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import streamlit as st

def plot_time_domain_signals(df, baseline_df, sensor_columns):
    st.subheader("📉 Time-Domain Signal Comparison")

        # Create 2 rows, 1 column subplot (one below the other)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Baseline", "Current")
    )

    # Baseline signal (top)
    fig.add_trace(go.Scatter(
        y=baseline_df['sensor1_filtered'].values,
        mode='lines',
        name='Baseline',
        line=dict(color='green')
    ), row=1, col=1)

    # Current signal (bottom)
    fig.add_trace(go.Scatter(
        y=df['sensor2_filtered'].values,
        mode='lines',
        name='Current',
        line=dict(color='red')
    ), row=2, col=1)

    fig.update_layout(
        title_text=f"Time Domain Signal - sensor",
        height=500,
        showlegend=False,
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Sample Index", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)



def plot_fft_with_anomalies(df, fault_results, sensor_columns):
    st.subheader("🔍 Frequency Spectrum (FFT) with Anomaly Markers")
    faulty_sensors = [f['sensor'] for f in fault_results] if fault_results else []

    for sensor in sensor_columns:
        freq_col = f"{sensor}_fftfreq"
        amp_col = f"{sensor}_fftspectrum"

        if freq_col not in df or amp_col not in df:
            st.warning(f"FFT data missing for {sensor}")
            continue

        freq = df[freq_col].values
        amp = df[amp_col].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freq, y=np.log1p(amp), name='FFT', line=dict(color='blue')))

        if sensor in faulty_sensors:
            max_index = np.argmax(amp)
            fig.add_trace(go.Scatter(
                x=[freq[max_index]],
                y=[np.log1p(amp[max_index])],
                mode='markers+text',
                text='⚠ Anomaly',
                textposition='top center',
                marker=dict(size=10, color='red'),
                name='Anomaly'
            ))

        fig.update_layout(
            title=f"FFT Spectrum - {sensor}",
            xaxis_title='Frequency (Hz)',
            yaxis_title='Log Amplitude',
            template='plotly_white',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_overlay_comparison(no_fault_df, current_df):
    st.subheader("🔁 Overlay Comparison with Baseline")
    common_cols = [col for col in current_df.columns if col in no_fault_df.columns]

    for col in common_cols[:3]:
        x1 = no_fault_df[col].dropna().values
        x2 = current_df[col].dropna().values
        min_len = min(len(x1), len(x2))

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=x1[:min_len], name='No Fault', line=dict(color='#FFE5B4')))
        fig.add_trace(go.Scatter(y=x2[:min_len], name='Current', line=dict(color='#0000FF')))
        fig.update_layout(
            title=f"Overlay - {col}",
            xaxis_title='Sample Index',
            yaxis_title='Amplitude',
            template='plotly_white',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_spectrogram(df, sensor_columns):
    st.subheader("🎛 Spectrogram (STFT)")
    for sensor in sensor_columns:
        try:
            f, t, Sxx = spectrogram(df[sensor].dropna().values, fs=1000)
            fig = go.Figure(data=go.Heatmap(
                z=10 * np.log10(Sxx + 1e-10),
                x=t,
                y=f,
                colorscale='Viridis'))

            fig.update_layout(
                title=f"Spectrogram - {sensor}",
                xaxis_title='Time [sec]',
                yaxis_title='Frequency [Hz]',
                height=300,
                template='plotly_white')

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Spectrogram failed for {sensor}: {e}")
