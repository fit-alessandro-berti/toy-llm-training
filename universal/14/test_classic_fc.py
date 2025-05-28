import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor  # Ensure this is imported
import warnings
import traceback  # For detailed error printing

# --- Configuration ---
BASE_DATA_DIR = "../../data/time_series/2"
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")
PREPROCESSOR_LOAD_PATH = "foundation_multitask_preprocessor_v3_ema_updated.npz"
OUTPUT_CSV_CLASSIC_FILENAME = "output_forecast_classic.csv"

DEFAULT_SEQ_LEN = 64
DEFAULT_PRED_HORIZONS = [1, 3, 5]

HGBR_PARAMS = {
    "max_iter": 100,
    "learning_rate": 0.1,
    "max_depth": 7,
    "l2_regularization": 0.1,
    "random_state": 42,
}
MIN_SAMPLES_FOR_TRAINING = 50


def create_windows_for_sensor(series, seq_len, pred_horizons, sensor_name):
    X, y = [], []
    max_horizon = 0
    if pred_horizons:  # Check if pred_horizons is not empty
        max_horizon = max(pred_horizons)
    else:  # Should not happen based on prior checks, but as a safeguard
        return np.array(X).reshape(0, seq_len if seq_len > 0 else 0), np.array(y).reshape(0, 0)

    if len(series) < seq_len + max_horizon:
        return np.array(X).reshape(0, seq_len if seq_len > 0 else 0), np.array(y).reshape(0, len(pred_horizons))

    for i in range(len(series) - seq_len - max_horizon + 1):
        input_window = series[i: i + seq_len]

        target_values = []
        valid_target = True
        for h in pred_horizons:
            target_val_idx = i + seq_len + h - 1
            # Ensure target_val_idx is within bounds
            if target_val_idx >= len(series):
                valid_target = False
                break
            target_val = series[target_val_idx]
            if np.isnan(target_val):
                valid_target = False
                break
            target_values.append(target_val)

        if valid_target:
            X.append(input_window)
            y.append(target_values)

    if not X:
        return np.array(X).reshape(0, seq_len if seq_len > 0 else 0), np.array(y).reshape(0, len(pred_horizons))

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


def forecast_with_classical_models():
    print(f"--- Classical Model Forecasting Script ---")
    print(f"Loading data from: {VALID_DIR}")
    print(f"Loading preprocessor info from: {PREPROCESSOR_LOAD_PATH}")

    try:
        preprocessor_data = np.load(PREPROCESSOR_LOAD_PATH, allow_pickle=True)
        global_means = preprocessor_data['global_means']
        global_stds = preprocessor_data['global_stds']
        canonical_sensor_names_obj = preprocessor_data['canonical_sensor_names']
        if isinstance(canonical_sensor_names_obj, np.ndarray):
            canonical_sensor_names = list(map(str, canonical_sensor_names_obj))
        else:
            canonical_sensor_names = list(map(str, list(canonical_sensor_names_obj)))

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
    if len(PRED_HORIZONS) == 0:  # Explicit check
        print("ERROR: PRED_HORIZONS list has zero length. Cannot make predictions. Exiting.")
        return

    all_sensor_data_for_training = {name: {'X': [], 'y': []} for name in canonical_sensor_names}
    file_paths = sorted(glob.glob(os.path.join(VALID_DIR, "*.csv")))

    if not file_paths:
        print(f"ERROR: No CSV files found in {VALID_DIR}. Exiting.")
        return

    print(f"\nStep 1: Preparing training data from all {len(file_paths)} CSV files...")
    for fp_idx, filepath in enumerate(file_paths):
        # print(f"  Processing file {fp_idx+1}/{len(file_paths)}: {os.path.basename(filepath)}") # Less verbose
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"    Warning: Could not read file {filepath}: {e}. Skipping.")
            continue

        for s_idx, sensor_name in enumerate(canonical_sensor_names):
            if sensor_name in df.columns:
                series_raw = df[sensor_name].astype(float)
                mean = global_means[s_idx]
                std = global_stds[s_idx]
                series_norm = series_raw.copy()  # Default to raw if std is too small
                if std >= 1e-8:
                    series_norm = (series_raw - mean) / std

                X_sensor, y_sensor = create_windows_for_sensor(series_norm, SEQ_LEN, PRED_HORIZONS, sensor_name)

                if X_sensor.size > 0:
                    all_sensor_data_for_training[sensor_name]['X'].append(X_sensor)
                    all_sensor_data_for_training[sensor_name]['y'].append(y_sensor)
        if (fp_idx + 1) % 10 == 0:
            print(f"  Processed {fp_idx + 1}/{len(file_paths)} files for data prep.")

    print("\nStep 2: Training one model per sensor...")
    trained_models = {}
    for s_idx, sensor_name in enumerate(canonical_sensor_names):
        if not all_sensor_data_for_training[sensor_name]['X']:
            continue

        X_list = all_sensor_data_for_training[sensor_name]['X']
        y_list = all_sensor_data_for_training[sensor_name]['y']

        if not X_list or not y_list:
            continue

        try:
            X_train_sensor = np.concatenate(X_list, axis=0)
            y_train_sensor = np.concatenate(y_list, axis=0)
        except ValueError as e:  # Handles cases where lists might be empty despite earlier checks if X_sensor.size was 0 but list not empty
            print(
                f"  Sensor {sensor_name}: Error concatenating data (likely empty lists or inconsistent sub-array shapes): {e}. Skipping.")
            continue

        if X_train_sensor.shape[0] >= MIN_SAMPLES_FOR_TRAINING and y_train_sensor.shape[0] == X_train_sensor.shape[
            0] and y_train_sensor.ndim == 2 and y_train_sensor.shape[1] == len(PRED_HORIZONS):
            print(
                f"  Training model for sensor: {sensor_name} with X_shape: {X_train_sensor.shape}, y_shape: {y_train_sensor.shape}, y_dtype: {y_train_sensor.dtype}")

            base_estimator = HistGradientBoostingRegressor(**HGBR_PARAMS)
            # Explicitly use MultiOutputRegressor
            # n_jobs=-1 can speed up training for multiple targets if CPU cores are available
            # For debugging, you can remove n_jobs or set to 1
            model_to_fit = MultiOutputRegressor(base_estimator, n_jobs=-1)

            print(f"    Model object type for fitting: {type(model_to_fit)}")  # DEBUG PRINT

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_to_fit.fit(X_train_sensor, y_train_sensor)
                trained_models[sensor_name] = model_to_fit
                print(f"    Successfully trained model for {sensor_name}.")
            except Exception as e:
                print(f"    ERROR training model for {sensor_name}: {e}")
                # print("    Detailed traceback:")
                # traceback.print_exc() # Uncomment for full traceback if errors persist
        elif X_train_sensor.shape[0] < MIN_SAMPLES_FOR_TRAINING:
            print(
                f"  Sensor {sensor_name}: Insufficient samples ({X_train_sensor.shape[0]}) for training. Min required: {MIN_SAMPLES_FOR_TRAINING}. Skipping.")
        else:  # Problem with y_train_sensor shape or consistency
            print(
                f"  Sensor {sensor_name}: Skipping due to inconsistent X/y samples or y_shape issues. X_shape: {X_train_sensor.shape}, y_shape: {y_train_sensor.shape}, Expected y_cols: {len(PRED_HORIZONS)}")

    if not trained_models:
        print(
            "ERROR: No models were trained. Cannot proceed to forecasting. Check data preparation and training logs. Exiting.")
        return

    print("\nStep 3: Generating forecasts and preparing output CSV...")
    output_data_list = []
    max_pred_horizon = 0
    if PRED_HORIZONS:  # Should always be true by now
        max_pred_horizon = max(PRED_HORIZONS)

    for fp_idx, filepath in enumerate(file_paths):
        # print(f"  Predicting for file {fp_idx+1}/{len(file_paths)}: {os.path.basename(filepath)}") # Less verbose
        try:
            df_current_file = pd.read_csv(filepath)
        except Exception as e:
            print(f"    Warning: Could not read file {filepath} for prediction: {e}. Skipping.")
            continue

        num_predictable_windows = len(df_current_file) - SEQ_LEN - max_pred_horizon + 1
        if num_predictable_windows < 0: num_predictable_windows = 0

        for window_start_idx in range(num_predictable_windows):
            for s_idx, sensor_name in enumerate(canonical_sensor_names):
                if sensor_name in df_current_file.columns and sensor_name in trained_models:
                    model = trained_models[sensor_name]
                    current_X_raw = df_current_file[sensor_name].iloc[
                                    window_start_idx: window_start_idx + SEQ_LEN].values.astype(float)

                    if len(current_X_raw) != SEQ_LEN:  # Should not happen with correct slicing
                        continue

                    mean_s = global_means[s_idx]
                    std_s = global_stds[s_idx]
                    current_X_norm = current_X_raw.copy()
                    if std_s >= 1e-8:
                        current_X_norm = (current_X_raw - mean_s) / std_s

                    predicted_y_norm = model.predict(current_X_norm.reshape(1, -1))[0]

                    for h_idx, horizon_val in enumerate(PRED_HORIZONS):
                        pred_val_norm = predicted_y_norm[h_idx]
                        pred_val_denorm = pred_val_norm
                        if std_s >= 1e-8:
                            pred_val_denorm = pred_val_norm * std_s + mean_s

                        actual_val_denorm = np.nan
                        difference_denorm = np.nan
                        target_time_idx = window_start_idx + SEQ_LEN + horizon_val - 1

                        if target_time_idx < len(df_current_file):
                            actual_val_raw = df_current_file[sensor_name].iloc[target_time_idx]
                            if pd.notna(actual_val_raw):
                                actual_val_denorm = float(actual_val_raw)
                                difference_denorm = pred_val_denorm - actual_val_denorm

                        output_data_list.append({
                            'filepath': os.path.basename(filepath),
                            'window_start_idx': window_start_idx,
                            'sensor_name': sensor_name,
                            'sensor_model_idx': s_idx,
                            'horizon_steps': horizon_val,
                            'predicted_value_denormalized': pred_val_denorm,
                            'actual_value_denormalized': actual_val_denorm,
                            'difference_denormalized': difference_denorm
                        })
        if (fp_idx + 1) % 10 == 0:
            print(f"  Predicted for {fp_idx + 1}/{len(file_paths)} files.")

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
    if not os.path.exists(PREPROCESSOR_LOAD_PATH):
        print(f"CRITICAL ERROR: Preprocessor file '{PREPROCESSOR_LOAD_PATH}' not found. Exiting.")
        exit()
    if not os.path.exists(VALID_DIR) or not os.listdir(VALID_DIR):
        print(f"CRITICAL ERROR: Validation directory '{VALID_DIR}' does not exist or is empty. Exiting.")
        exit()

    forecast_with_classical_models()