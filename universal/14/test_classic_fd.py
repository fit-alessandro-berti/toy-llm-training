import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier  # <<< ADD THIS IMPORT
import warnings
import traceback

# --- Configuration ---
BASE_DATA_DIR = "../../data/time_series/2"
VALID_DIR = os.path.join(BASE_DATA_DIR, "VALIDATION")
PREPROCESSOR_LOAD_PATH = "foundation_multitask_preprocessor_v3_ema_updated.npz"
OUTPUT_CSV_CLASSIC_FAILURE_FILENAME = "output_failure_prediction_classic.csv"

DEFAULT_SEQ_LEN = 64
DEFAULT_FAIL_HORIZONS = [3, 5, 10]

HGBCLASS_PARAMS = {
    "max_iter": 100,
    "learning_rate": 0.1,
    "max_depth": 7,
    "l2_regularization": 0.1,
    "random_state": 42,
}
MIN_SAMPLES_FOR_TRAINING = 100


def create_failure_prediction_windows(df, seq_len, fail_horizons,
                                      canonical_sensor_names, global_means, global_stds,
                                      filepath_basename):
    X_all_windows = []
    Y_all_horizons = []
    window_info_list = []

    num_canonical_sensors = len(canonical_sensor_names)
    feature_vector_len = num_canonical_sensors * seq_len

    if 'CURRENT_FAILURE' not in df.columns:
        # print(f"Warning: 'CURRENT_FAILURE' column missing in {filepath_basename}. Skipping file for failure data.") # Less verbose
        return np.array(X_all_windows).reshape(0, feature_vector_len), \
            np.array(Y_all_horizons).reshape(0, len(fail_horizons) if fail_horizons else 0), \
            window_info_list

    failure_flags_series = df['CURRENT_FAILURE'].to_numpy()
    normalized_sensor_data = np.full((len(df), num_canonical_sensors), 0.0)  # Initialize with 0.0 (mean for normalized)

    for s_idx, s_name in enumerate(canonical_sensor_names):
        if s_name in df.columns:
            series_raw = df[s_name].astype(float).to_numpy()
            mean = global_means[s_idx]
            std = global_stds[s_idx]
            if std >= 1e-8:
                normalized_sensor_data[:, s_idx] = (series_raw - mean) / std
            # else: it remains 0.0 as initialized

    # NaNs from sensors not present in CSV but in canonical list, or from original NaNs in present sensors,
    # will become 0.0 due to np.nan_to_num or if std was ~0. HGB can handle NaNs if they were preserved.
    # For simplicity here, they become 0.0.
    normalized_sensor_data = np.nan_to_num(normalized_sensor_data, nan=0.0)

    max_horizon = 0
    if fail_horizons:
        max_horizon = max(fail_horizons)
    else:
        return np.array(X_all_windows).reshape(0, feature_vector_len), \
            np.array(Y_all_horizons).reshape(0, 0), \
            window_info_list

    for i in range(len(df) - seq_len - max_horizon + 1):
        window_sensor_data = normalized_sensor_data[i: i + seq_len, :]
        feature_vector = window_sensor_data.flatten()
        X_all_windows.append(feature_vector)

        targets_for_window = []
        for fh in fail_horizons:
            start_idx_target = i + seq_len
            end_idx_target = i + seq_len + fh

            if end_idx_target > len(failure_flags_series):  # Ensure slice is within bounds
                # This should ideally not be hit if outer loop range is correct based on max_horizon
                targets_for_window.append(0)  # Or handle as error/skip window
                # print(f"Warning: Target window out of bounds for file {filepath_basename}, window_start {i}, horizon {fh}")
                continue  # This will lead to fewer targets than fail_horizons for this sample, might need robust handling or ensure loop is correct

            target_window_failure_flags = failure_flags_series[start_idx_target:end_idx_target]

            if np.any(target_window_failure_flags == 1):
                targets_for_window.append(1)
            else:
                targets_for_window.append(0)

        # Ensure targets_for_window has the correct number of elements
        if len(targets_for_window) == len(fail_horizons):
            Y_all_horizons.append(targets_for_window)
            window_info_list.append({
                'filepath': filepath_basename,
                'window_start_idx': i
            })
        else:  # If not all targets could be formed, discard this X window too
            X_all_windows.pop()

    if not X_all_windows:
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
        fail_horizons_np = preprocessor_data.get('fail_horizons', np.array(DEFAULT_FAIL_HORIZONS))
        FAIL_HORIZONS = list(map(int, fail_horizons_np)) if isinstance(fail_horizons_np,
                                                                       np.ndarray) else DEFAULT_FAIL_HORIZONS

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

        if X_file.size > 0 and Y_file.size > 0 and X_file.shape[0] == Y_file.shape[
            0]:  # Ensure X and Y have same num samples
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
    print(
        f"  Training with X_shape: {X_train_global.shape}, Y_shape: {Y_train_global.shape} (dtype: {Y_train_global.dtype})")

    # --- MODIFICATION: Use MultiOutputClassifier ---
    base_classifier = HistGradientBoostingClassifier(**HGBCLASS_PARAMS)
    model_to_fit = MultiOutputClassifier(base_classifier,
                                         n_jobs=-1)  # n_jobs=-1 for parallel training if multiple targets
    # --- END MODIFICATION ---

    trained_model = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(f"  Fitting with model type: {type(model_to_fit)}")  # Confirm wrapper is used
            model_to_fit.fit(X_train_global, Y_train_global)  # Y_train_global is (n_samples, n_outputs)
        trained_model = model_to_fit
        print(f"  Successfully trained global failure prediction model.")
    except Exception as e:
        print(f"    ERROR training global failure prediction model: {e}")
        traceback.print_exc()  # Print full traceback
        return

    print("\nStep 3: Generating failure predictions and preparing output CSV...")
    output_data_list = []

    for fp_idx, filepath in enumerate(file_paths):
        try:
            df_current_file = pd.read_csv(filepath)
        except Exception as e:
            print(f"    Warning: Could not read file {filepath} for prediction: {e}. Skipping.")
            continue

        X_file_pred, Y_file_gt, window_info_file = create_failure_prediction_windows(
            df_current_file, SEQ_LEN, FAIL_HORIZONS, canonical_sensor_names,
            global_means, global_stds, os.path.basename(filepath)
        )

        if X_file_pred.size == 0 or Y_file_gt.size == 0 or X_file_pred.shape[0] != Y_file_gt.shape[0]:
            continue

        predicted_labels_batch = trained_model.predict(X_file_pred)
        predicted_probas_list_of_arrays = trained_model.predict_proba(X_file_pred)

        for i in range(X_file_pred.shape[0]):
            window_info = window_info_file[i]

            for fh_idx, failure_horizon_val in enumerate(FAIL_HORIZONS):
                gt_label = int(Y_file_gt[i, fh_idx])
                pred_label = int(predicted_labels_batch[i, fh_idx])

                # predicted_probas_list_of_arrays is a list of (n_samples, n_classes=2) arrays
                # We need the probability for class '1' for the fh_idx-th output (horizon)
                try:
                    pred_proba_class1 = float(predicted_probas_list_of_arrays[fh_idx][i, 1])
                except IndexError:
                    print(f"IndexError accessing predicted_probas_list_of_arrays.")
                    print(f"fh_idx: {fh_idx}, len(list): {len(predicted_probas_list_of_arrays)}")
                    if fh_idx < len(predicted_probas_list_of_arrays):
                        print(
                            f"Shape of proba array at fh_idx: {predicted_probas_list_of_arrays[fh_idx].shape}, sample_idx i: {i}")
                    pred_proba_class1 = 0.0  # Fallback, should not happen with correct MultiOutputClassifier usage

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
