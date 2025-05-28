import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier  # Classifier version
import warnings
import traceback

# --- Configuration ---
BASE_DATA_DIR = "../../data/time_series/1"
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")
PREPROCESSOR_LOAD_PATH = "foundation_multitask_preprocessor_v3_ema_updated.npz"
OUTPUT_CSV_CLASSIC_FAILURE_FILENAME = "output_failure_prediction_classic.csv"

DEFAULT_SEQ_LEN = 64
DEFAULT_FAIL_HORIZONS = [3, 5, 10]  # Horizons for failure prediction

HGBCLASS_PARAMS = {  # Parameters for HistGradientBoostingClassifier
    "max_iter": 100,
    "learning_rate": 0.1,
    "max_depth": 7,
    "l2_regularization": 0.1,
    "random_state": 42,
    # Loss defaults to 'log_loss' which is appropriate for binary/multiclass
}
MIN_SAMPLES_FOR_TRAINING = 100  # Min windows to train the global failure model


def create_failure_prediction_windows(df, seq_len, fail_horizons,
                                      canonical_sensor_names, global_means, global_stds,
                                      filepath_basename):
    """
    Creates input features (X) and multi-horizon failure targets (Y) from a DataFrame.
    X: Normalized, flattened sensor data for seq_len.
    Y: Binary indicators for failure within each horizon.
    Returns X, Y, and a list of window_info dicts for traceability.
    """
    X_all_windows = []
    Y_all_horizons = []
    window_info_list = []

    num_canonical_sensors = len(canonical_sensor_names)
    feature_vector_len = num_canonical_sensors * seq_len

    # Ensure 'CURRENT_FAILURE' column exists
    if 'CURRENT_FAILURE' not in df.columns:
        print(
            f"Warning: 'CURRENT_FAILURE' column missing in {filepath_basename}. Cannot generate failure targets for this file.")
        return np.array(X_all_windows).reshape(0, feature_vector_len), \
            np.array(Y_all_horizons).reshape(0, len(fail_horizons)), \
            window_info_list

    failure_flags_series = df['CURRENT_FAILURE'].to_numpy()

    # Pre-normalize all relevant sensor columns
    normalized_sensor_data = np.full((len(df), num_canonical_sensors), np.nan)  # Initialize with NaN

    for s_idx, s_name in enumerate(canonical_sensor_names):
        if s_name in df.columns:
            series_raw = df[s_name].astype(float).to_numpy()
            mean = global_means[s_idx]
            std = global_stds[s_idx]
            if std >= 1e-8:
                normalized_sensor_data[:, s_idx] = (series_raw - mean) / std
            else:  # If std is near zero, normalized data is 0 (if mean was subtracted) or raw (if not)
                normalized_sensor_data[:, s_idx] = 0.0  # Represent as mean (0 after std normalization)

    # Fill any remaining NaNs in normalized_sensor_data (e.g. sensors not in file) with 0
    # This means they are at their 'global mean' for the feature vector
    normalized_sensor_data = np.nan_to_num(normalized_sensor_data, nan=0.0)

    max_horizon = 0
    if fail_horizons:
        max_horizon = max(fail_horizons)
    else:  # Should be caught by earlier checks
        return np.array(X_all_windows).reshape(0, feature_vector_len), \
            np.array(Y_all_horizons).reshape(0, 0), \
            window_info_list

    for i in range(len(df) - seq_len - max_horizon + 1):
        # Create feature vector X for the window
        # Slices from normalized_sensor_data which has shape (len(df), num_canonical_sensors)
        # We want a slice of shape (seq_len, num_canonical_sensors) then flatten
        window_sensor_data = normalized_sensor_data[i: i + seq_len, :]
        feature_vector = window_sensor_data.flatten()  # Shape: (seq_len * num_canonical_sensors)
        X_all_windows.append(feature_vector)

        # Create target vector Y for failure horizons
        targets_for_window = []
        for fh in fail_horizons:
            # Check for failure in interval [i + seq_len, i + seq_len + fh -1] (inclusive indices)
            # This corresponds to (t, t + fh] where t is the end of the input window
            start_idx_target = i + seq_len
            end_idx_target = i + seq_len + fh  # Exclusive end for slicing: [start, end)

            # Ensure target window is within bounds of failure_flags_series
            if end_idx_target > len(failure_flags_series):
                # This window cannot have its full targets defined, skip (loop range should prevent this for y)
                # This case should be rare if loop range is correct: len(df) - seq_len - max_horizon + 1
                # This means we might not be able to form a full target vector.
                # For simplicity in this function, we assume the loop range correctly handles this.
                # If an error occurs, it suggests an off-by-one in loop range or max_horizon.
                pass  # This target cannot be formed

            target_window_failure_flags = failure_flags_series[start_idx_target:end_idx_target]

            if np.any(target_window_failure_flags == 1):
                targets_for_window.append(1)
            else:
                targets_for_window.append(0)
        Y_all_horizons.append(targets_for_window)

        window_info_list.append({
            'filepath': filepath_basename,
            'window_start_idx': i
        })

    if not X_all_windows:  # If list is empty
        return np.array(X_all_windows).reshape(0, feature_vector_len), \
            np.array(Y_all_horizons).reshape(0, len(fail_horizons) if fail_horizons else 0), \
            window_info_list

    return np.array(X_all_windows, dtype=np.float64), \
        np.array(Y_all_horizons, dtype=np.int64), \
        window_info_list


def test_classical_failure_prediction():
    print(f"--- Classical Model Failure Prediction Script ---")
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
        # Load FAIL_HORIZONS, not PRED_HORIZONS for this script's primary task
        fail_horizons_np = preprocessor_data.get('fail_horizons', np.array(DEFAULT_FAIL_HORIZONS))
        FAIL_HORIZONS = list(map(int, fail_horizons_np)) if isinstance(fail_horizons_np,
                                                                       np.ndarray) else DEFAULT_FAIL_HORIZONS

        # model_max_sensors_dim = int(preprocessor_data.get('model_max_sensors_dim', len(canonical_sensor_names)))
        # num_features = model_max_sensors_dim * SEQ_LEN

        print(f"Using SEQ_LEN={SEQ_LEN}, FAIL_HORIZONS={FAIL_HORIZONS}")
        print(f"Found {len(canonical_sensor_names)} canonical sensors.")

    except FileNotFoundError:
        print(f"ERROR: Preprocessor file not found at {PREPROCESSOR_LOAD_PATH}. Exiting.")
        return
    except KeyError as e:
        print(f"ERROR: Missing key {e} in preprocessor file. Exiting.")
        return
    if not FAIL_HORIZONS or len(FAIL_HORIZONS) == 0:
        print("ERROR: FAIL_HORIZONS is empty or not defined. Cannot make predictions. Exiting.")
        return

    all_X_for_training_list = []
    all_Y_for_training_list = []

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

        X_file, Y_file, _ = create_failure_prediction_windows(
            df, SEQ_LEN, FAIL_HORIZONS, canonical_sensor_names, global_means, global_stds, os.path.basename(filepath)
        )

        if X_file.size > 0 and Y_file.size > 0:
            all_X_for_training_list.append(X_file)
            all_Y_for_training_list.append(Y_file)

        if (fp_idx + 1) % 20 == 0 or (fp_idx + 1) == len(file_paths):
            print(f"  Processed {fp_idx + 1}/{len(file_paths)} files for data preparation.")

    if not all_X_for_training_list or not all_Y_for_training_list:
        print("ERROR: No training data could be prepared from any file. Exiting.")
        return

    X_train_global = np.concatenate(all_X_for_training_list, axis=0)
    Y_train_global = np.concatenate(all_Y_for_training_list, axis=0)

    if X_train_global.shape[0] < MIN_SAMPLES_FOR_TRAINING:
        print(
            f"ERROR: Insufficient total samples ({X_train_global.shape[0]}) for training. Min required: {MIN_SAMPLES_FOR_TRAINING}. Exiting.")
        return
    if Y_train_global.shape[0] != X_train_global.shape[0] or Y_train_global.ndim != 2 or Y_train_global.shape[1] != len(
            FAIL_HORIZONS):
        print(
            f"ERROR: Mismatch in training data shapes. X_shape: {X_train_global.shape}, Y_shape: {Y_train_global.shape}, Expected Y_cols: {len(FAIL_HORIZONS)}. Exiting.")
        return

    print(f"\nStep 2: Training a global failure prediction model...")
    print(f"  Training with X_shape: {X_train_global.shape}, Y_shape: {Y_train_global.shape}")

    model = HistGradientBoostingClassifier(**HGBCLASS_PARAMS)
    trained_model = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # HistGradientBoostingClassifier supports multi-output Y if it's 2D (n_samples, n_outputs)
            # It will fit one estimator per output.
            model.fit(X_train_global, Y_train_global)
        trained_model = model
        print(f"  Successfully trained global failure prediction model.")
    except Exception as e:
        print(f"    ERROR training global failure prediction model: {e}")
        traceback.print_exc()
        return  # Cannot proceed without a model

    print("\nStep 3: Generating failure predictions and preparing output CSV...")
    output_data_list = []

    for fp_idx, filepath in enumerate(file_paths):
        try:
            df_current_file = pd.read_csv(filepath)
        except Exception as e:
            print(f"    Warning: Could not read file {filepath} for prediction: {e}. Skipping.")
            continue

        # Get X, ground truth Y, and window_info for the current file
        X_file_pred, Y_file_gt, window_info_file = create_failure_prediction_windows(
            df_current_file, SEQ_LEN, FAIL_HORIZONS, canonical_sensor_names,
            global_means, global_stds, os.path.basename(filepath)
        )

        if X_file_pred.size == 0 or Y_file_gt.size == 0:  # No windows to predict for this file
            continue

        # Predict labels and probabilities
        predicted_labels_batch = trained_model.predict(X_file_pred)  # Shape: (n_windows, n_fail_horizons)

        # predict_proba returns a list of arrays for multi-output: one per output/horizon
        # Each array is (n_samples, n_classes=2 for binary)
        predicted_probas_list_of_arrays = trained_model.predict_proba(X_file_pred)

        for i in range(X_file_pred.shape[0]):  # For each window in the file
            window_info = window_info_file[i]

            for fh_idx, failure_horizon_val in enumerate(FAIL_HORIZONS):
                gt_label = int(Y_file_gt[i, fh_idx])
                pred_label = int(predicted_labels_batch[i, fh_idx])

                # Probability of class '1' (failure) for the fh_idx-th horizon
                # The i-th sample, for the fh_idx-th output, probability of class 1
                pred_proba_class1 = float(predicted_probas_list_of_arrays[fh_idx][i, 1])

                output_data_list.append({
                    'filepath': window_info['filepath'],
                    'window_start_idx': window_info['window_start_idx'],
                    'failure_horizon_steps': failure_horizon_val,
                    'ground_truth_failure': gt_label,
                    'predicted_failure_probability': pred_proba_class1,
                    'predicted_failure_label': pred_label
                })

        if (fp_idx + 1) % 20 == 0 or (fp_idx + 1) == len(file_paths):
            print(f"  Predicted for {fp_idx + 1}/{len(file_paths)} files.")

    if output_data_list:
        output_df = pd.DataFrame(output_data_list)
        try:
            output_df.to_csv(OUTPUT_CSV_CLASSIC_FAILURE_FILENAME, index=False)
            print(
                f"\nSuccessfully wrote {len(output_df)} failure prediction data points to {OUTPUT_CSV_CLASSIC_FAILURE_FILENAME}")
        except Exception as e:
            print(f"\nERROR: Could not write failure prediction outputs to CSV: {e}")
    else:
        print("\nNo failure prediction data generated to write to CSV.")

    print("\n--- Classical Model Failure Prediction Script Finished ---")


if __name__ == '__main__':
    if not os.path.exists(PREPROCESSOR_LOAD_PATH):
        print(f"CRITICAL ERROR: Preprocessor file '{PREPROCESSOR_LOAD_PATH}' not found. Exiting.")
        exit()
    if not os.path.exists(VALID_DIR) or not os.listdir(VALID_DIR):
        print(f"CRITICAL ERROR: Validation directory '{VALID_DIR}' does not exist or is empty. Exiting.")
        exit()

    test_classical_failure_prediction()
