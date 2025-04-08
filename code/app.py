import streamlit as st
import os
import threading
import time
import pandas as pd
from data_loader import get_available_machines, get_machine_and_data, load_data
from signal_processing import process_signals,get_signal_results
from fault_detection import detect_fault
from fault_interpreter import interpret_fault
from graph_utils import (
    plot_fft_with_anomalies,
    plot_time_domain_signals,
    plot_overlay_comparison,
    plot_spectrogram,
    live_data_plot,
)

DATA_DIR = "machine_data"
st.set_page_config(page_title="Vibration Analysis", layout="wide")
st.title("🛠️ Real-Time Vibration Analysis System")

# Step 1: Machine selection
machines = get_available_machines()
machine_options = machines + ["➕ Add new machine"]
selected_machine = st.selectbox("Select Machine", machine_options)

# Step 2: New machine setup
if selected_machine == "➕ Add new machine":
    new_machine_name = st.text_input("Enter machine name")
    if new_machine_name:
        no_fault_file = st.file_uploader(f"Upload 'No Fault' data for '{new_machine_name}'", type=['csv'])
        if no_fault_file:
            path = os.path.join(DATA_DIR, f"{new_machine_name}_no_fault.csv")
            with open(path, "wb") as f:
                f.write(no_fault_file.read())
            st.success(f"Saved no fault data for '{new_machine_name}'. Please select it from the list.")
            st.stop()
    else:
        st.warning("Enter a machine name.")
        st.stop()

# Step 3: Load no fault data for selected machine
_, no_fault_df, _ = get_machine_and_data(selected_machine)
if no_fault_df is None:
    st.error("❌ Failed to load no fault data.")
    st.stop()
st.success("✅ No fault data loaded.")

# Step 4: Upload current data
uploaded_file = st.file_uploader("📅 Upload current vibration data for analysis", type=["csv"])
if uploaded_file is None:
    st.stop()

current_df = load_data(uploaded_file)
if current_df is None:
    st.error("❌ Failed to load current data.")
    st.stop()

st.success("✅ Current data loaded successfully.")

sensor_columns = [col for col in current_df.columns if col.lower().startswith("sensor")]

# Live Graph Start
if st.button("Start Live View"):
    live_data_plot(current_df, sensor_columns)

# Step 5: Signal processing
st.markdown("### 🧪 Signal Processing & Fault Detection")
no_fault_processed = process_signals(no_fault_df, sensor_columns)
current_processed = process_signals(current_df, sensor_columns)

# For additional results like spectrum/statistics
no_fault_results = get_signal_results(no_fault_df, sensor_columns)
current_results = get_signal_results(current_df, sensor_columns)

st.dataframe(current_processed.head(10))
st.dataframe(no_fault_processed.head(10))

# Step 6: Fault detection
fault_results = detect_fault(current_processed, no_fault_processed, sensor_columns)

# Step 7: Visualization toggles
st.markdown("### 📊 Graphical Analysis")
with st.expander("📉 Time Domain Signals"):
    plot_time_domain_signals(current_processed , no_fault_processed, sensor_columns)

with st.expander("🔍 FFT with Anomaly Detection"):
    plot_fft_with_anomalies(current_processed, fault_results, sensor_columns)

with st.expander("🔁 Overlay Comparison with No-Fault Data"):
    plot_overlay_comparison(no_fault_df, current_df)

with st.expander("🎛 Spectrogram View (STFT)"):
    plot_spectrogram(current_df, sensor_columns)

# Step 8: Final Decision Display
st.markdown("---")
st.markdown("## 🧠 Final Machine Health Evaluation")

if fault_results:
    st.error("🚨 FAULT DETECTED!")
    for report in fault_results:
        sensor = report['sensor']
        msg = report['message']
        deviation = report['deviation']

        st.markdown(f"🔴 **{sensor}**: {msg}")
        interpreted_fault = interpret_fault(current_results,no_fault_results)
        st.markdown(f"🛠️ **Likely Defect**: `{interpreted_fault}`")

        with st.expander(f"📊 View Deviation Details for {sensor}"):
            st.write(f"**RMS**: Current = {deviation['RMS'][0]:.4f}, Baseline = {deviation['RMS'][1]:.4f}")
            st.write(f"**Crest Factor**: Current = {deviation['Crest'][0]:.4f}, Baseline = {deviation['Crest'][1]:.4f}")
            st.write(f"**Kurtosis**: Current = {deviation['Kurtosis'][0]:.4f}, Baseline = {deviation['Kurtosis'][1]:.4f}")

    st.toast("⚠️ Fault detected in machine!", icon="🚨")
else:
    st.success("✅ Machine is in PERFECT CONDITION. No anomalies detected.")
    st.toast("🎉 All good! No faults found.", icon="✅")
