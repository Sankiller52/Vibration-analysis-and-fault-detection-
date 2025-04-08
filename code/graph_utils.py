import streamlit as st
import plotly.graph_objs as go
import numpy as np
from scipy.signal import spectrogram

# Live Data Plot (continuously updating, pinned to top)
def live_data_plot(df, sensor_columns):
    st.markdown("<div style='position:fixed; top:70px; right:20px; z-index:9999; background-color:#f9f9f9; padding:10px; border-radius:10px; box-shadow:0px 0px 10px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
    st.markdown("**📡 Live Monitoring (Unprocessed Data)**")

    for sensor in sensor_columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df[sensor].values,
            mode='lines',
            name=sensor,
            line=dict(color='royalblue')
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title='Sample Index',
            yaxis_title='Amplitude',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def plot_time_domain_signals(df, baseline_df, sensor_columns):
    st.subheader("📉 Time-Domain Signal Comparison")
    for sensor in sensor_columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=baseline_df[f'{sensor}_filtered'].values, name='Baseline', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(y=df[f'{sensor}_filtered'].values, name='Current', line=dict(color='red')))

        fig.update_layout(
            title=f"Time Domain - {sensor}",
            xaxis_title='Sample Index',
            yaxis_title='Amplitude',
            template='plotly_white',
            height=300,
            showlegend=True
        )
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
        fig.add_trace(go.Scatter(y=x1[:min_len], name='No Fault', line=dict(color='green')))
        fig.add_trace(go.Scatter(y=x2[:min_len], name='Current', line=dict(color='red')))
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
