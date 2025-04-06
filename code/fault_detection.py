import numpy as np

# --- Compare extracted features with baseline ---
def detect_fault(live_df, baseline_df, sensor_columns, threshold=0.15):
    fault_report = []
    for sensor in sensor_columns:
        faults = []

        # Get features from current and baseline
        live_rms = live_df[f'{sensor}_rms'].mean()
        base_rms = baseline_df[f'{sensor}_rms'].mean()

        live_crest = live_df[f'{sensor}_crest'].mean()
        base_crest = baseline_df[f'{sensor}_crest'].mean()

        live_kurt = live_df[f'{sensor}_kurtosis'].mean()
        base_kurt = baseline_df[f'{sensor}_kurtosis'].mean()

        # Simple threshold comparison
        if abs(live_rms - base_rms) / base_rms > threshold:
            faults.append("RMS deviation")

        if abs(live_crest - base_crest) / base_crest > threshold:
            faults.append("Crest Factor anomaly")

        if abs(live_kurt - base_kurt) / base_kurt > threshold:
            faults.append("Kurtosis irregularity")

        if faults:
            fault_report.append({
                'sensor': sensor,
                'message': f"Fault detected in {sensor}: " + ", ".join(faults),
                'deviation': {
                    'RMS': (live_rms, base_rms),
                    'Crest': (live_crest, base_crest),
                    'Kurtosis': (live_kurt, base_kurt)
                }
            })

    return fault_report
