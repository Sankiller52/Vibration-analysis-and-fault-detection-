import os
import pandas as pd

DATA_DIR = "machine_data"

def load_data(file_obj_or_path):
    """
    Loads and preprocesses CSV data from a file path or file-like object.
    """
    try:
        # Handle uploaded file (Streamlit)
        if hasattr(file_obj_or_path, 'read'):
            df = pd.read_csv(file_obj_or_path)

        # Handle file path
        elif isinstance(file_obj_or_path, (str, bytes, os.PathLike)):
            if not os.path.exists(file_obj_or_path):
                print(f"[ERROR] File not found: {file_obj_or_path}")
                return None
            df = pd.read_csv(file_obj_or_path)
        else:
            print(f"[ERROR] Unsupported input type: {type(file_obj_or_path)}")
            return None

        # Clean and preprocess
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df.dropna(how='all', inplace=True)

        if not any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
            print("[ERROR] No numeric data found in file.")
            return None

        return df

    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None


def get_available_machines():
    """
    Lists all machines with no-fault data.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    files = os.listdir(DATA_DIR)
    machines = [f.replace("_no_fault.csv", "") for f in files if f.endswith("_no_fault.csv")]
    return machines


def get_machine_and_data(selected_machine):
    """
    Tries to load the no-fault data for a machine.
    Returns:
        - available machines
        - DataFrame of no-fault data (or None)
        - True if it's a new machine, False if known
    """
    machines = get_available_machines()

    # ✅ Automatically create machine folder if missing
    machine_folder = os.path.join(DATA_DIR, selected_machine)
    if not os.path.exists(machine_folder):
        os.makedirs(machine_folder)

    if selected_machine in machines:
        file_path = os.path.join(DATA_DIR, f"{selected_machine}_no_fault.csv")
        df = load_data(file_path)
        return machines, df, False
    else:
        return machines, None, True
