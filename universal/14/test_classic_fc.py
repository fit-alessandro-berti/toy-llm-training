import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings
import traceback  # For detailed error printing

# --- Configuration ---
BASE_DATA_DIR = "../../data/time_series/1"  # Example, adjust if needed
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")  # Source of TestData
PREPROCESSOR_LOAD_PATH = "foundation_multitask_preprocessor_v3_ema_updated.npz"  # From DL script
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


def create_windows_for_sensor(series_np, seq_len, pred_horizons, sensor_name):  # Expects series_np as numpy array
    """
    Creates input windows (X) and multi-horizon target vectors (y) for a single sensor series.
    NaNs in X will be kept (HGBR can handle them). Windows with NaN targets are skipped.
    """
    X, y = [], []
    max_horizon = 0
    if not pred_horizons:
        return np.array(X).reshape(0, seq_len if seq_len > 0 else 0), np.array(y).reshape(0, 0)
    max_horizon = max(pred_horizons)

    if len(series_np) < seq_len + max_horizon:
        return np.array(X).reshape(0, seq_len if seq_len > 0 else 0), np.array(y).reshape(0, len(pred_horizons))

    for i in range(len(series_np) - seq_len - max_horizon + 1):
        input_window = series_np[i: i + seq_len]

        target_values = []
        valid_target = True
        for h_val in pred_horizons:
            target_val_idx = i + seq_len + h_val - 1
            if target_val_idx >= len(series_np):
                valid_target = False
                break
            target_val = series_np[target_val_idx]
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
        print(f"ERROR: Missing key {e} in preprocessor file. Exiting.")
        return
    if not PRED_HORIZONS or len(PRED_HORIZONS) == 0:
        print("ERROR: PRED_HORIZONS is empty or not defined. Cannot make predictions. Exiting.")
        return

    all_sensor_data_for_training = {name: {'X': [], 'y': []} for name in canonical_sensor_names}
    file_paths = sorted(glob.glob(os.path.join(VALID_DIR, "*.csv")))

    if not file_paths:
        print(f"ERROR: No CSV files found in {VALID_DIR}. Exiting.")
        return

    print(f"\nStep 1: Preparing training data from all {len(file_paths)} CSV files...")
    for fp_idx, filepath in enumerate(file_paths):
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"    Warning: Could not read file {filepath}: {e}. Skipping.")
            continue

        for s_idx, sensor_name in enumerate(canonical_sensor_names):
            if sensor_name in df.columns:
                series_raw = df[sensor_name].astype(float)  # Pandas Series
                mean = global_means[s_idx]
                std = global_stds[s_idx]
                series_norm_pd = series_raw.copy()  # Keep as Pandas Series for now
                if std >= 1e-8:
                    series_norm_pd = (series_raw - mean) / std

                # create_windows_for_sensor expects a numpy array
                X_sensor, y_sensor = create_windows_for_sensor(series_norm_pd.to_numpy(), SEQ_LEN, PRED_HORIZONS,
                                                               sensor_name)

                if X_sensor.size > 0:
                    all_sensor_data_for_training[sensor_name]['X'].append(X_sensor)
                    all_sensor_data_for_training[sensor_name]['y'].append(y_sensor)
        if (fp_idx + 1) % 20 == 0 or (fp_idx + 1) == len(file_paths):
            print(f"  Processed {fp_idx + 1}/{len(file_paths)} files for data preparation.")

    print("\nStep 2: Training one model per sensor...")
    trained_models = {}
    for s_idx, sensor_name in enumerate(canonical_sensor_names):
        if not all_sensor_data_for_training[sensor_name]['X']:
            continue
        X_list = all_sensor_data_for_training[sensor_name]['X']
        y_list = all_sensor_data_for_training[sensor_name]['y']
        if not X_list or not y_list: continue

        try:
            X_train_sensor = np.concatenate(X_list, axis=0)
            y_train_sensor = np.concatenate(y_list, axis=0)
        except ValueError as e:
            print(f"  Sensor {sensor_name}: Error concatenating training data: {e}. Skipping.")
            continue

        if X_train_sensor.shape[0] >= MIN_SAMPLES_FOR_TRAINING and \
                y_train_sensor.shape[0] == X_train_sensor.shape[0] and \
                y_train_sensor.ndim == 2 and y_train_sensor.shape[1] == len(PRED_HORIZONS):
            print(
                f"  Training model for sensor: {sensor_name} with X_shape: {X_train_sensor.shape}, y_shape: {y_train_sensor.shape}")
            base_estimator = HistGradientBoostingRegressor(**HGBR_PARAMS)
            model_to_fit = MultiOutputRegressor(base_estimator, n_jobs=-1)  # n_jobs=-1 for parallel fitting of targets

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_to_fit.fit(X_train_sensor, y_train_sensor)
                trained_models[sensor_name] = model_to_fit
            except Exception as e:
                print(f"    ERROR training model for {sensor_name}: {e}")
        elif X_train_sensor.shape[0] < MIN_SAMPLES_FOR_TRAINING:
            print(
                f"  Sensor {sensor_name}: Insufficient samples ({X_train_sensor.shape[0]}) for training. Min required: {MIN_SAMPLES_FOR_TRAINING}. Skipping.")
        else:
            print(
                f"  Sensor {sensor_name}: Skipping training due to inconsistent X/y samples or y_shape issues. X_shape: {X_train_sensor.shape}, y_shape: {y_train_sensor.shape}, Expected y_cols: {len(PRED_HORIZONS)}")

    if not trained_models:
        print("ERROR: No models were trained. Cannot proceed to forecasting. Exiting.")
        return

    print("\nStep 3: Generating forecasts (Optimized Prediction Phase)...")
    output_data_list = []
    max_pred_horizon = max(PRED_HORIZONS) if PRED_HORIZONS else 0

    for fp_idx, filepath in enumerate(file_paths):
        try:
            df_current_file = pd.read_csv(filepath)
        except Exception as e:
            print(f"    Warning: Could not read file {filepath} for prediction: {e}. Skipping.")
            continue

        for s_idx, sensor_name in enumerate(canonical_sensor_names):
            if sensor_name not in df_current_file.columns or sensor_name not in trained_models:
                continue

            model = trained_models[sensor_name]
            series_raw_np = df_current_file[sensor_name].astype(float).to_numpy()  # Get full series as numpy array

            mean_s = global_means[s_idx]
            std_s = global_stds[s_idx]
            series_norm_np = series_raw_np.copy()
            if std_s >= 1e-8:
                series_norm_np = (series_raw_np - mean_s) / std_s

            # Determine the number of windows for which full predictions (including all targets) can be made
            # This is the number of start indices for which a full SEQ_LEN window and all its targets exist
            num_predictable_windows = len(series_norm_np) - SEQ_LEN - max_pred_horizon + 1
            if num_predictable_windows <= 0:  # If not enough data to form even one full window + targets
                continue

            # Use sliding_window_view to create all X input windows efficiently
            # Input must be a NumPy array.
            # The shape of `all_possible_X_input_windows` will be (total_possible_start_points, SEQ_LEN)
            if len(series_norm_np) < SEQ_LEN:  # Not enough data to form any input window
                continue
            all_possible_X_input_windows = np.lib.stride_tricks.sliding_window_view(series_norm_np,
                                                                                    window_shape=SEQ_LEN)

            # We only need to predict for windows where all targets can be fetched
            X_batch_to_predict = all_possible_X_input_windows[:num_predictable_windows]

            if X_batch_to_predict.shape[0] == 0:  # If no valid windows to predict
                continue

            # Perform batch prediction
            all_predicted_y_norm_batch = model.predict(
                X_batch_to_predict)  # Shape: (num_predictable_windows, num_horizons)

            # Iterate through the predictions and their corresponding start indices
            for i in range(num_predictable_windows):
                window_start_idx = i  # The index 'i' corresponds to the window start in series_raw_np/series_norm_np
                predicted_y_norm_single_window = all_predicted_y_norm_batch[i]  # Predictions for this window

                for h_idx, horizon_val in enumerate(PRED_HORIZONS):
                    pred_val_norm = predicted_y_norm_single_window[h_idx]

                    pred_val_denorm = pred_val_norm
                    if std_s >= 1e-8:  # De-normalize
                        pred_val_denorm = pred_val_norm * std_s + mean_s

                    actual_val_denorm = np.nan
                    difference_denorm = np.nan
                    target_time_idx = window_start_idx + SEQ_LEN + horizon_val - 1

                    # series_raw_np is the original, unnormalized data for the sensor for the current file
                    if target_time_idx < len(series_raw_np):  # Ensure target index is within bounds
                        actual_val_raw_val = series_raw_np[target_time_idx]
                        if pd.notna(actual_val_raw_val):  # Check if actual is not NaN
                            actual_val_denorm = float(actual_val_raw_val)
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
        if (fp_idx + 1) % 20 == 0 or (fp_idx + 1) == len(file_paths):
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