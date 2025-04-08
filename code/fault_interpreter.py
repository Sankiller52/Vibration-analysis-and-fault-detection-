# fault_interpreter.py

def interpret_fault(current_results, baseline_results, thresholds=None):
    """
    Compare current and baseline sensor features and classify possible gearbox faults.

    Args:
        current_results (dict): From process_signals() -> current features.
        baseline_results (dict): From process_signals() -> baseline features.
        thresholds (dict, optional): Custom thresholds for fault classification.

    Returns:
        List of dicts: Each dict contains sensor name, message, and feature deviation.
    """

    if thresholds is None:
        # Default thresholds (can be adjusted as needed)
        thresholds = {
            'RMS': 1.2,         # 20% increase
            'CrestFactor': 1.2,
            'Kurtosis': 1.5     # 50% increase
        }

    fault_reports = []

    for sensor in current_results:
        current = current_results[sensor]['features']
        baseline = baseline_results[sensor]['features']

        deviation = {
            'RMS': (current['RMS'], baseline['RMS']),
            'Crest': (current['CrestFactor'], baseline['CrestFactor']),
            'Kurtosis': (current['Kurtosis'], baseline['Kurtosis'])
        }

        msg = "❌ Fault Detected – Type: "

        # Determine if thresholds are exceeded
        fault_detected = False

        if deviation['Kurtosis'][0] > deviation['Kurtosis'][1] * thresholds['Kurtosis']:
            msg += "Misalignment / Gear Tooth Crack"
            fault_detected = True
        elif deviation['Crest'][0] > deviation['Crest'][1] * thresholds['CrestFactor']:
            msg += "Eccentricity or Local Defect"
            fault_detected = True
        elif deviation['RMS'][0] > deviation['RMS'][1] * thresholds['RMS']:
            msg += "Unbalance or Looseness"
            fault_detected = True

        if fault_detected:
            fault_reports.append({
                'sensor': sensor,
                'message': msg,
                'deviation': deviation
            })
        else:
            fault_reports.append({
                'sensor': sensor,
                'message': "⚠️ Fault Detected – Defect Unknown",
                'deviation': deviation
            })

    return fault_reports
