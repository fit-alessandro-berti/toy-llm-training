import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
import traceback

# --- Configuration ---
BASE_DATA_DIR = "../../data/time_series/1"
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")
PREPROCESSOR_LOAD_PATH = "foundation_multitask_preprocessor_v3_ema_updated.npz"
OUTPUT_CSV_CLASSIC_RCA_FILENAME = "output_rca_classic.csv"

DEFAULT_SEQ_LEN = 64
DEFAULT_RCA_FAILURE_LOOKAHEAD = 3  # Default if not in preprocessor

HGBCLASS_PARAMS_RCA = {  # Parameters for per-sensor RCA classifiers
    "max_iter": 50,  # Potentially many models, keep iterations lower for speed
    "learning_rate": 0.1,
    "max_depth": 5,
    "l2_regularization": 0.1,
    "random_state": 42,
    "min_samples_leaf": 20,  # To prevent overfitting on smaller class if imbalance
}
MIN_SAMPLES_FOR_RCA_TRAINING = 50  # Min windows for training each sensor's RCA model
MIN_POSITIVE_SAMPLES_FOR_RCA_TRAINING = 10  # Min positive RCA labels for a sensor to train its model


def create_rca_input_output_windows(df, seq_len, rca_lookahead,
                                    canonical_sensor_names, global_means, global_stds,
                                    filepath_basename):
    """
    Creates:
    X_global_features: Normalized, flattened global sensor data windows.
    Y_per_sensor_rca_labels: Ground truth RCA labels (0/1) for each sensor for each window.
    Y_overall_failure_context: Flag (0/1) if any failure occurred in the RCA lookahead window.
    window_info_list: For traceability.
    """
    X_global_list = []
    Y_per_sensor_rca_list = []
    Y_overall_failure_context_list = []
    window_info_out_list = []

    num_canonical_sensors = len(canonical_sensor_names)
    feature_vector_len = num_canonical_sensors * seq_len

    if 'CURRENT_FAILURE' not in df.columns:
        return np.array(X_global_list).reshape(0, feature_vector_len), \
            np.array(Y_per_sensor_rca_list).reshape(0, num_canonical_sensors), \
            np.array(Y_overall_failure_context_list), \
            window_info_out_list

    failure_flags_all = df['CURRENT_FAILURE'].to_numpy()

    # Pre-normalize all canonical sensor data from the file
    normalized_sensor_data_file = np.full((len(df), num_canonical_sensors), 0.0)  # Default to 0 (mean)
    raw_sensor_data_file = np.full((len(df), num_canonical_sensors), np.nan)  # For RCA GT calculation

    for s_idx, s_name in enumerate(canonical_sensor_names):
        if s_name in df.columns:
            series_raw = df[s_name].astype(float).to_numpy()
            raw_sensor_data_file[:, s_idx] = series_raw  # Store raw for GT

            mean = global_means[s_idx]
            std = global_stds[s_idx]
            if std >= 1e-8:
                normalized_sensor_data_file[:, s_idx] = (series_raw - mean) / std
            # else: it remains 0.0 (mean for normalized features)

    # Fill NaNs in normalized data (e.g., sensors not in CSV) with 0.0
    normalized_sensor_data_file = np.nan_to_num(normalized_sensor_data_file, nan=0.0)

    # Max lookahead is just rca_lookahead for this function's purpose
    for i in range(len(df) - seq_len - rca_lookahead + 1):
        # 1. Create global X feature vector for the window
        window_all_sensors_norm_data = normalized_sensor_data_file[i: i + seq_len, :]
        global_feature_vector = window_all_sensors_norm_data.flatten()
        X_global_list.append(global_feature_vector)

        # 2. Determine overall failure context for the RCA window
        rca_window_start_idx = i + seq_len
        rca_window_end_idx = i + seq_len + rca_lookahead

        overall_failure_in_rca_win = 0
        if rca_window_end_idx <= len(failure_flags_all):  # Ensure window is within bounds
            failure_flags_in_rca_win = failure_flags_all[rca_window_start_idx:rca_window_end_idx]
            if np.any(failure_flags_in_rca_win == 1):
                overall_failure_in_rca_win = 1
        Y_overall_failure_context_list.append(overall_failure_in_rca_win)

        # 3. Determine per-sensor RCA ground truth labels
        sensor_rca_labels_for_window = np.zeros(num_canonical_sensors, dtype=np.int64)
        if overall_failure_in_rca_win == 1 and rca_lookahead > 0:  # Only calculate RCA if failure present
            current_input_window_raw = raw_sensor_data_file[i: i + seq_len, :]
            future_rca_window_raw = raw_sensor_data_file[rca_window_start_idx:rca_window_end_idx, :]

            for s_idx in range(num_canonical_sensors):
                # Get non-NaN data for current and future windows for this sensor
                sensor_data_current_input_raw = current_input_window_raw[:, s_idx]
                sensor_data_current_input_raw = sensor_data_current_input_raw[~np.isnan(sensor_data_current_input_raw)]

                sensor_data_future_rca_raw = future_rca_window_raw[:, s_idx]
                sensor_data_future_rca_raw = sensor_data_future_rca_raw[~np.isnan(sensor_data_future_rca_raw)]

                if len(sensor_data_current_input_raw) > 0 and len(sensor_data_future_rca_raw) > 0:
                    mean_current = np.mean(sensor_data_current_input_raw)
                    std_current = np.std(sensor_data_current_input_raw)
                    std_current = max(std_current, 1e-6)  # Avoid division by zero

                    # Check if any point in the future window deviates significantly
                    if np.any(np.abs(sensor_data_future_rca_raw - mean_current) > 3 * std_current):
                        sensor_rca_labels_for_window[s_idx] = 1
        Y_per_sensor_rca_list.append(sensor_rca_labels_for_window)

        window_info_out_list.append({
            'filepath': filepath_basename,
            'window_start_idx': i
        })

    if not X_global_list:  # If list is empty
        return np.array(X_global_list).reshape(0, feature_vector_len), \
            np.array(Y_per_sensor_rca_list).reshape(0, num_canonical_sensors), \
            np.array(Y_overall_failure_context_list), \
            window_info_out_list

    return np.array(X_global_list, dtype=np.float64), \
        np.array(Y_per_sensor_rca_list, dtype=np.int64), \
        np.array(Y_overall_failure_context_list, dtype=np.int64), \
        window_info_out_list


def test_classical_rca():
    print(f"--- Classical Model RCA Script ---")
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
        # RCA_FAILURE_LOOKAHEAD is the key horizon for this script
        # Use fail_horizons from preprocessor to determine a default rca_failure_lookahead if needed.
        fail_horizons_np = preprocessor_data.get('fail_horizons', np.array([]))  # Allow empty
        fail_horizons_list = list(map(int, fail_horizons_np)) if isinstance(fail_horizons_np,
                                                                            np.ndarray) and fail_horizons_np.size > 0 else []

        default_rca_lh = fail_horizons_list[0] if fail_horizons_list else DEFAULT_RCA_FAILURE_LOOKAHEAD
        RCA_LOOKAHEAD = int(preprocessor_data.get('rca_failure_lookahead', default_rca_lh))

        print(f"Using SEQ_LEN={SEQ_LEN}, RCA_FAILURE_LOOKAHEAD={RCA_LOOKAHEAD}")
        print(f"Found {len(canonical_sensor_names)} canonical sensors.")

    except FileNotFoundError:
        print(f"ERROR: Preprocessor file '{PREPROCESSOR_LOAD_PATH}' not found. Exiting."); return
    except KeyError as e:
        print(f"ERROR: Missing key {e} in preprocessor file. Exiting."); return
    if RCA_LOOKAHEAD <= 0: print("ERROR: RCA_FAILURE_LOOKAHEAD must be > 0. Exiting."); return

    # --- Step 1: Data Preparation ---
    all_X_global_list = []
    all_Y_per_sensor_rca_list = []
    # We don't need to store all_Y_overall_failure_context_list for training, it's re-derived for output

    file_paths = sorted(glob.glob(os.path.join(VALID_DIR, "*.csv")))
    if not file_paths: print(f"ERROR: No CSV files found in {VALID_DIR}. Exiting."); return

    print(f"\nStep 1: Preparing training data for RCA from all {len(file_paths)} CSV files...")
    for fp_idx, filepath in enumerate(file_paths):
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"    Warning: Could not read file {filepath}: {e}. Skipping."); continue

        X_file, Y_rca_file, _, _ = create_rca_input_output_windows(
            df, SEQ_LEN, RCA_LOOKAHEAD, canonical_sensor_names,
            global_means, global_stds, os.path.basename(filepath)
        )

        if X_file.size > 0 and Y_rca_file.size > 0 and X_file.shape[0] == Y_rca_file.shape[0]:
            all_X_global_list.append(X_file)
            all_Y_per_sensor_rca_list.append(Y_rca_file)

        if (fp_idx + 1) % 20 == 0 or (fp_idx + 1) == len(file_paths):
            print(f"  Processed {fp_idx + 1}/{len(file_paths)} files for data prep.")

    if not all_X_global_list: print("ERROR: No training data (X) could be prepared. Exiting."); return
    if not all_Y_per_sensor_rca_list: print("ERROR: No RCA target data (Y) could be prepared. Exiting."); return

    X_global_train = np.concatenate(all_X_global_list, axis=0)
    Y_per_sensor_rca_train_all = np.concatenate(all_Y_per_sensor_rca_list, axis=0)

    if X_global_train.shape[0] < MIN_SAMPLES_FOR_RCA_TRAINING:
        print(
            f"ERROR: Insufficient total samples ({X_global_train.shape[0]}) for training. Min: {MIN_SAMPLES_FOR_RCA_TRAINING}. Exiting.");
        return
    if Y_per_sensor_rca_train_all.shape[0] != X_global_train.shape[0] or \
            Y_per_sensor_rca_train_all.shape[1] != len(canonical_sensor_names):
        print(
            f"ERROR: Mismatch in aggregated X and Y_rca shapes. X: {X_global_train.shape}, Y_rca: {Y_per_sensor_rca_train_all.shape}. Exiting.");
        return

    # --- Step 2: Training one RCA classifier per sensor ---
    print(f"\nStep 2: Training one RCA classifier per sensor...")
    trained_rca_sensor_models = {}
    for s_idx, sensor_name in enumerate(canonical_sensor_names):
        Y_rca_for_this_sensor = Y_per_sensor_rca_train_all[:, s_idx]  # Target for this sensor's RCA model

        num_positive_labels = np.sum(Y_rca_for_this_sensor == 1)
        if X_global_train.shape[0] >= MIN_SAMPLES_FOR_RCA_TRAINING and \
                num_positive_labels >= MIN_POSITIVE_SAMPLES_FOR_RCA_TRAINING and \
                (X_global_train.shape[
                     0] - num_positive_labels) >= MIN_POSITIVE_SAMPLES_FOR_RCA_TRAINING:  # Ensure some negative samples too

            print(
                f"  Training RCA model for sensor: {sensor_name} ({s_idx + 1}/{len(canonical_sensor_names)}) with {X_global_train.shape[0]} samples ({num_positive_labels} positive RCA labels).")
            model = HistGradientBoostingClassifier(**HGBCLASS_PARAMS_RCA)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_global_train, Y_rca_for_this_sensor)  # Y is 1D here
                trained_rca_sensor_models[sensor_name] = model
            except Exception as e:
                print(f"    ERROR training RCA model for sensor {sensor_name}: {e}")
                # traceback.print_exc()
        else:
            print(
                f"  Skipping RCA model training for sensor {sensor_name}: Insufficient samples or positive/negative labels. Total: {X_global_train.shape[0]}, Positives: {num_positive_labels}.")

    if not trained_rca_sensor_models:
        print("ERROR: No RCA models were successfully trained. Cannot proceed. Exiting.")
        return
    print(f"  Trained RCA models for {len(trained_rca_sensor_models)} sensors.")

    # --- Step 3: Generating RCA predictions and preparing output CSV ---
    print("\nStep 3: Generating classical RCA predictions...")
    output_data_list = []

    for fp_idx, filepath in enumerate(file_paths):
        try:
            df_current_file = pd.read_csv(filepath)
        except Exception as e:
            print(f"    Warning: Could not read file {filepath} for prediction: {e}. Skipping."); continue

        X_file_pred, Y_rca_file_gt, Y_fail_context_file, window_info_file = create_rca_input_output_windows(
            df_current_file, SEQ_LEN, RCA_LOOKAHEAD, canonical_sensor_names,
            global_means, global_stds, os.path.basename(filepath)
        )

        if X_file_pred.size == 0: continue

        for i in range(X_file_pred.shape[0]):  # For each window in the file
            current_X_window_features = X_file_pred[i, :].reshape(1, -1)  # Reshape for single sample prediction
            window_info = window_info_file[i]
            gt_overall_failure_in_win = Y_fail_context_file[i]

            for s_idx, sensor_name in enumerate(canonical_sensor_names):
                pred_rca_label = 0  # Default if model not available or sensor inactive for RCA
                pred_rca_proba = 0.0

                if sensor_name in trained_rca_sensor_models:
                    sensor_rca_model = trained_rca_sensor_models[sensor_name]
                    try:
                        pred_rca_label = int(sensor_rca_model.predict(current_X_window_features)[0])
                        # predict_proba gives [[prob_class_0, prob_class_1]] for single sample
                        pred_rca_proba = float(sensor_rca_model.predict_proba(current_X_window_features)[0, 1])
                    except Exception as e:
                        print(f"Error predicting with RCA model for {sensor_name} on window {window_info}: {e}")

                gt_sensor_rca_label = int(Y_rca_file_gt[i, s_idx])

                output_data_list.append({
                    'filepath': window_info['filepath'],
                    'window_start_idx': window_info['window_start_idx'],
                    'rca_lookahead_horizon_config': RCA_LOOKAHEAD,
                    'gt_overall_failure_in_rca_window': gt_overall_failure_in_win,
                    'sensor_name': sensor_name,
                    'sensor_model_idx': s_idx,
                    'ground_truth_sensor_rca_label': gt_sensor_rca_label,
                    'predicted_sensor_rca_label': pred_rca_label,
                    'predicted_sensor_rca_probability': pred_rca_proba
                })

        if (fp_idx + 1) % 20 == 0 or (fp_idx + 1) == len(file_paths):
            print(f"  Predicted RCA for {fp_idx + 1}/{len(file_paths)} files.")

    if output_data_list:
        output_df = pd.DataFrame(output_data_list)
        try:
            output_df.to_csv(OUTPUT_CSV_CLASSIC_RCA_FILENAME, index=False)
            print(
                f"\nSuccessfully wrote {len(output_df)} classical RCA data points to {OUTPUT_CSV_CLASSIC_RCA_FILENAME}")
        except Exception as e:
            print(f"\nERROR: Could not write classical RCA outputs to CSV: {e}")
    else:
        print("\nNo classical RCA data generated to write to CSV.")

    print("\n--- Classical Model RCA Script Finished ---")


if __name__ == '__main__':
    if not os.path.exists(PREPROCESSOR_LOAD_PATH): print(
        f"CRITICAL ERROR: Preprocessor file '{PREPROCESSOR_LOAD_PATH}' not found. Exiting."); exit()
    if not os.path.exists(VALID_DIR) or not os.listdir(VALID_DIR): print(
        f"CRITICAL ERROR: Validation directory '{VALID_DIR}' does not exist or is empty. Exiting."); exit()
    test_classical_rca()
