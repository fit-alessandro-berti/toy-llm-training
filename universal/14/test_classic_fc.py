import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split  # Not used for final training, but good import
import warnings

# --- Configuration ---
# Data paths (adjust as in your previous script)
BASE_DATA_DIR = "../../data/time_series/1"
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")
PREPROCESSOR_LOAD_PATH = "foundation_multitask_preprocessor_v3_ema_updated.npz"  # From DL script
OUTPUT_CSV_CLASSIC_FILENAME = "output_forecast_classic.csv"

# Parameters (should match the context of your data and DL model for comparison)
# These will be loaded from preprocessor if available, otherwise defaults used.
DEFAULT_SEQ_LEN = 64
DEFAULT_PRED_HORIZONS = [1, 3, 5]  # e.g., predict 1, 3, 5 steps ahead

# Model parameters for HistGradientBoostingRegressor
HGBR_PARAMS = {
    "max_iter": 100,  # Number of boosting iterations
    "learning_rate": 0.1,
    "max_depth": 7,  # Max depth of each tree
    "l2_regularization": 0.1,
    "random_state": 42,
    # HistGradientBoostingRegressor handles NaNs by default
}

MIN_SAMPLES_FOR_TRAINING = 50  # Minimum number of windows required to train a model for a sensor


def create_windows_for_sensor(series, seq_len, pred_horizons, sensor_name):
    """
    Creates input windows (X) and multi-horizon target vectors (y) for a single sensor series.
    NaNs in X will be kept (HGBR can handle them). Windows with NaN targets are skipped.
    """
    X, y = [], []
    max_horizon = max(pred_horizons)
    # Ensure series is a numpy array
    if isinstance(series, pd.Series):
        series = series.to_numpy()

    if len(series) < seq_len + max_horizon:
        # print(f"Warning: Series for sensor {sensor_name} is too short ({len(series)}) for seq_len {seq_len} and max_horizon {max_horizon}. Skipping.")
        return np.array(X), np.array(y)  # Return empty arrays with proper shape for consistency

    for i in range(len(series) - seq_len - max_horizon + 1):
        input_window = series[i: i + seq_len]

        target_values = []
        valid_target = True
        for h in pred_horizons:
            target_val = series[i + seq_len + h - 1]
            if np.isnan(target_val):
                valid_target = False
                break
            target_values.append(target_val)

        if valid_target:
            X.append(input_window)
            y.append(target_values)

    if not X:  # if X is empty
        return np.array(X).reshape(0, seq_len if seq_len > 0 else 0), np.array(y).reshape(0,
                                                                                          len(pred_horizons) if pred_horizons else 0)

    return np.array(X), np.array(y)


def forecast_with_classical_models():
    print(f"--- Classical Model Forecasting Script ---")
    print(f"Loading data from: {VALID_DIR}")
    print(f"Loading preprocessor info from: {PREPROCESSOR_LOAD_PATH}")

    # 1. Load Preprocessor Config (means, stds, sensor names, sequence/horizon params)
    try:
        preprocessor_data = np.load(PREPROCESSOR_LOAD_PATH, allow_pickle=True)
        global_means = preprocessor_data['global_means']
        global_stds = preprocessor_data['global_stds']
        # Ensure canonical_sensor_names is a list of strings
        canonical_sensor_names_obj = preprocessor_data['canonical_sensor_names']
        if isinstance(canonical_sensor_names_obj, np.ndarray):
            canonical_sensor_names = list(map(str, canonical_sensor_names_obj))
        else:  # Assuming it's already a list-like of strings or can be converted
            canonical_sensor_names = list(map(str, list(canonical_sensor_names_obj)))

        # Load sequence and horizon parameters from preprocessor, with fallbacks
        SEQ_LEN = int(preprocessor_data.get('seq_len', DEFAULT_SEQ_LEN))
        pred_horizons_np = preprocessor_data.get('pred_horizons', np.array(DEFAULT_PRED_HORIZONS))
        PRED_HORIZONS = list(map(int, pred_horizons_np)) if isinstance(pred_horizons_np,
                                                                       np.ndarray) else DEFAULT_PRED_HORIZONS

        print(f"Using SEQ_LEN={SEQ_LEN}, PRED_HORIZONS={PRED_HORIZONS}")
        print(f"Found {len(canonical_sensor_names)} canonical sensors in preprocessor.")

    except FileNotFoundError:
        print(f"ERROR: Preprocessor file not found at {PREPROCESSOR_LOAD_PATH}. Exiting.")
        return
    except KeyError as e:
        print(
            f"ERROR: Missing key {e} in preprocessor file. Ensure it contains 'global_means', 'global_stds', 'canonical_sensor_names', 'seq_len', 'pred_horizons'. Exiting.")
        return
    if not PRED_HORIZONS:
        print("ERROR: PRED_HORIZONS is empty. Cannot make predictions. Exiting.")
        return

    all_sensor_data_for_training = {name: {'X': [], 'y': []} for name in canonical_sensor_names}
    file_paths = sorted(glob.glob(os.path.join(VALID_DIR, "*.csv")))  # Sort for deterministic order

    if not file_paths:
        print(f"ERROR: No CSV files found in {VALID_DIR}. Exiting.")
        return

    print(f"\nStep 1: Preparing training data from all {len(file_paths)} CSV files...")
    for fp_idx, filepath in enumerate(file_paths):
        print(f"  Processing file {fp_idx + 1}/{len(file_paths)}: {os.path.basename(filepath)}")
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"    Warning: Could not read file {filepath}: {e}. Skipping.")
            continue

        for s_idx, sensor_name in enumerate(canonical_sensor_names):
            if sensor_name in df.columns:
                series_raw = df[sensor_name].astype(float)  # Ensure numeric type

                mean = global_means[s_idx]
                std = global_stds[s_idx]

                if std < 1e-8:  # Avoid division by zero or near-zero std
                    # print(f"    Sensor {sensor_name} has std near zero ({std}). Using raw values or skipping normalization.")
                    series_norm = series_raw.copy()  # Or handle as constant
                else:
                    series_norm = (series_raw - mean) / std

                X_sensor, y_sensor = create_windows_for_sensor(series_norm, SEQ_LEN, PRED_HORIZONS, sensor_name)

                if X_sensor.size > 0:  # Check if any windows were created
                    all_sensor_data_for_training[sensor_name]['X'].append(X_sensor)
                    all_sensor_data_for_training[sensor_name]['y'].append(y_sensor)

    print("\nStep 2: Training one model per sensor...")
    trained_models = {}
    for s_idx, sensor_name in enumerate(canonical_sensor_names):
        if not all_sensor_data_for_training[sensor_name]['X']:  # No data collected
            # print(f"  Sensor {sensor_name}: No training data windows collected. Skipping model training.")
            continue

        X_list = all_sensor_data_for_training[sensor_name]['X']
        y_list = all_sensor_data_for_training[sensor_name]['y']

        # Ensure X_list and y_list are not empty before concatenation
        if not X_list or not y_list:
            # print(f"  Sensor {sensor_name}: Training data lists are empty after file processing. Skipping model training.")
            continue

        try:
            X_train_sensor = np.concatenate(X_list, axis=0)
            y_train_sensor = np.concatenate(y_list, axis=0)
        except ValueError as e:
            print(
                f"  Sensor {sensor_name}: Error concatenating data (likely empty or inconsistent shapes): {e}. Skipping.")
            continue

        if X_train_sensor.shape[0] >= MIN_SAMPLES_FOR_TRAINING:
            print(f"  Training model for sensor: {sensor_name} with {X_train_sensor.shape[0]} samples.")
            model = HistGradientBoostingRegressor(**HGBR_PARAMS)
            try:
                with warnings.catch_warnings():  # Suppress potential convergence warnings for demo
                    warnings.simplefilter("ignore")
                    model.fit(X_train_sensor, y_train_sensor)
                trained_models[sensor_name] = model
            except Exception as e:
                print(f"    ERROR training model for {sensor_name}: {e}")
        else:
            print(
                f"  Sensor {sensor_name}: Insufficient samples ({X_train_sensor.shape[0]}) for training. Min required: {MIN_SAMPLES_FOR_TRAINING}. Skipping.")

    if not trained_models:
        print(
            "ERROR: No models were trained. Cannot proceed to forecasting. Check data and MIN_SAMPLES_FOR_TRAINING. Exiting.")
        return

    print("\nStep 3: Generating forecasts and preparing output CSV...")
    output_data_list = []
    max_pred_horizon = max(PRED_HORIZONS)

    for fp_idx, filepath in enumerate(file_paths):  # Iterate through files again for prediction
        print(f"  Predicting for file {fp_idx + 1}/{len(file_paths)}: {os.path.basename(filepath)}")
        try:
            df_current_file = pd.read_csv(filepath)
        except Exception as e:
            print(f"    Warning: Could not read file {filepath} for prediction: {e}. Skipping.")
            continue

        # Max possible start index for a window to have all targets within the file
        num_predictable_windows = len(df_current_file) - SEQ_LEN - max_pred_horizon + 1

        for window_start_idx in range(num_predictable_windows):
            for s_idx, sensor_name in enumerate(canonical_sensor_names):
                if sensor_name in df_current_file.columns and sensor_name in trained_models:
                    model = trained_models[sensor_name]

                    # Extract current input window (raw, then normalize)
                    current_X_raw = df_current_file[sensor_name].iloc[
                                    window_start_idx: window_start_idx + SEQ_LEN].values.astype(float)

                    mean_s = global_means[s_idx]
                    std_s = global_stds[s_idx]

                    if std_s < 1e-8:
                        current_X_norm = current_X_raw.copy()  # Or handle as constant
                    else:
                        current_X_norm = (current_X_raw - mean_s) / std_s

                    # HGBR handles NaNs in input, so explicit imputation might not be needed if default behavior is acceptable.
                    # If any value is NaN, prediction might be affected. For consistency, let's ensure it's a valid float array.
                    # current_X_norm[np.isnan(current_X_norm)] = 0 # Example: impute with 0 (mean of std normal)
                    # HGBR handles NaNs, so this specific line can be omitted.

                    if current_X_norm.shape[
                        0] != SEQ_LEN:  # Should not happen with iloc slicing if num_predictable_windows is correct
                        # print(f"Warning: Incorrect window shape for {sensor_name} in {os.path.basename(filepath)} at index {window_start_idx}. Skipping prediction.")
                        continue

                    predicted_y_norm = model.predict(current_X_norm.reshape(1, -1))[0]  # Predict expects 2D

                    for h_idx, horizon_val in enumerate(PRED_HORIZONS):
                        pred_val_norm = predicted_y_norm[h_idx]

                        # De-normalize prediction
                        if std_s < 1e-8:
                            pred_val_denorm = pred_val_norm  # Value was not normalized if std was ~0
                        else:
                            pred_val_denorm = pred_val_norm * std_s + mean_s

                        # Get actual future value (raw, already de-normalized)
                        actual_val_denorm = np.nan
                        difference_denorm = np.nan

                        target_time_idx = window_start_idx + SEQ_LEN + horizon_val - 1
                        if target_time_idx < len(df_current_file):
                            actual_val_raw = df_current_file[sensor_name].iloc[target_time_idx]
                            if pd.notna(actual_val_raw):  # Check if not NaN
                                actual_val_denorm = float(actual_val_raw)
                                difference_denorm = pred_val_denorm - actual_val_denorm

                        output_data_list.append({
                            'filepath': os.path.basename(filepath),
                            'window_start_idx': window_start_idx,
                            'sensor_name': sensor_name,
                            'sensor_model_idx': s_idx,  # Index in canonical_sensor_names
                            'horizon_steps': horizon_val,
                            'predicted_value_denormalized': pred_val_denorm,
                            'actual_value_denormalized': actual_val_denorm,
                            'difference_denormalized': difference_denorm
                        })

    # Step 4: Write to CSV
    if output_data_list:
        output_df = pd.DataFrame(output_data_list)
        try:
            output_df.to_csv(OUTPUT_CSV_CLASSIC_FILENAME, index=False)
            print(f"\nSuccessfully wrote {len(output_df)} forecast data points to {OUTPUT_CSV_CLASSIC_FILENAME}")
        except Exception as e:
            print(f"\nERROR: Could not write forecast outputs to CSV: {e}")
    else:
        print("\nNo forecast data generated to write to CSV.")

    print("\n--- Classical Model Forecasting Script Finished ---")


if __name__ == '__main__':
    # Basic path checks
    if not os.path.exists(PREPROCESSOR_LOAD_PATH):
        print(
            f"CRITICAL ERROR: Preprocessor file '{PREPROCESSOR_LOAD_PATH}' not found. This file is essential. Exiting.")
        exit()
    if not os.path.exists(VALID_DIR) or not os.listdir(VALID_DIR):
        print(
            f"CRITICAL ERROR: Validation directory '{VALID_DIR}' does not exist or is empty. No data to process. Exiting.")
        exit()

    forecast_with_classical_models()
