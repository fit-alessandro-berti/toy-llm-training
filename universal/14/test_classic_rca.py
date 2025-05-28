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
DEFAULT_RCA_FAILURE_LOOKAHEAD = 3

HGBCLASS_PARAMS_RCA = {
    "max_iter": 50,
    "learning_rate": 0.1,
    "max_depth": 5,
    "l2_regularization": 0.1,
    "random_state": 42,
    "min_samples_leaf": 20,
}
MIN_SAMPLES_FOR_RCA_TRAINING = 50
MIN_POSITIVE_SAMPLES_FOR_RCA_TRAINING = 10


def create_rca_input_output_windows(df, seq_len, rca_lookahead,
                                    canonical_sensor_names, global_means, global_stds,
                                    filepath_basename):
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
    normalized_sensor_data_file = np.full((len(df), num_canonical_sensors), 0.0)
    raw_sensor_data_file = np.full((len(df), num_canonical_sensors), np.nan)

    for s_idx, s_name in enumerate(canonical_sensor_names):
        if s_name in df.columns:
            series_raw = df[s_name].astype(float).to_numpy()
            raw_sensor_data_file[:, s_idx] = series_raw
            mean = global_means[s_idx]
            std = global_stds[s_idx]
            if std >= 1e-8:
                normalized_sensor_data_file[:, s_idx] = (series_raw - mean) / std
    normalized_sensor_data_file = np.nan_to_num(normalized_sensor_data_file, nan=0.0)

    # The loop range ensures that rca_window_end_idx will be within bounds
    for i in range(len(df) - seq_len - rca_lookahead + 1):
        window_all_sensors_norm_data = normalized_sensor_data_file[i: i + seq_len, :]
        global_feature_vector = window_all_sensors_norm_data.flatten()
        X_global_list.append(global_feature_vector)

        rca_window_start_idx = i + seq_len
        rca_window_end_idx = i + seq_len + rca_lookahead
        overall_failure_in_rca_win = 0
        failure_flags_in_rca_win = failure_flags_all[rca_window_start_idx:rca_window_end_idx]
        if np.any(failure_flags_in_rca_win == 1):
            overall_failure_in_rca_win = 1
        Y_overall_failure_context_list.append(overall_failure_in_rca_win)

        sensor_rca_labels_for_window = np.zeros(num_canonical_sensors, dtype=np.int64)
        if overall_failure_in_rca_win == 1 and rca_lookahead > 0:
            current_input_window_raw = raw_sensor_data_file[i: i + seq_len, :]
            future_rca_window_raw = raw_sensor_data_file[rca_window_start_idx:rca_window_end_idx, :]
            for s_idx in range(num_canonical_sensors):
                sensor_data_current_input_raw = current_input_window_raw[:, s_idx]
                sensor_data_current_input_raw = sensor_data_current_input_raw[~np.isnan(sensor_data_current_input_raw)]
                sensor_data_future_rca_raw = future_rca_window_raw[:, s_idx]
                sensor_data_future_rca_raw = sensor_data_future_rca_raw[~np.isnan(sensor_data_future_rca_raw)]

                if len(sensor_data_current_input_raw) > 0 and len(sensor_data_future_rca_raw) > 0:
                    mean_current = np.mean(sensor_data_current_input_raw)
                    std_current = np.std(sensor_data_current_input_raw)
                    std_current = max(std_current, 1e-6)
                    if np.any(np.abs(sensor_data_future_rca_raw - mean_current) > 3 * std_current):
                        sensor_rca_labels_for_window[s_idx] = 1
        Y_per_sensor_rca_list.append(sensor_rca_labels_for_window)
        window_info_out_list.append({'filepath': filepath_basename, 'window_start_idx': i})

    if not X_global_list:
        return np.array(X_global_list).reshape(0, feature_vector_len), \
            np.array(Y_per_sensor_rca_list).reshape(0, num_canonical_sensors), \
            np.array(Y_overall_failure_context_list), window_info_out_list
    return np.array(X_global_list, dtype=np.float64), \
        np.array(Y_per_sensor_rca_list, dtype=np.int64), \
        np.array(Y_overall_failure_context_list, dtype=np.int64), window_info_out_list


def test_classical_rca():
    print(f"--- Classical Model RCA Script ---")
    print(f"Loading data from: {VALID_DIR}")
    print(f"Loading preprocessor info from: {PREPROCESSOR_LOAD_PATH}")

    try:
        preprocessor_data = np.load(PREPROCESSOR_LOAD_PATH, allow_pickle=True)
        global_means = preprocessor_data['global_means']
        global_stds = preprocessor_data['global_stds']
        csn_obj = preprocessor_data['canonical_sensor_names']
        canonical_sensor_names = list(map(str, csn_obj)) if isinstance(csn_obj, np.ndarray) else list(
            map(str, list(csn_obj)))
        SEQ_LEN = int(preprocessor_data.get('seq_len', DEFAULT_SEQ_LEN))
        fh_np = preprocessor_data.get('fail_horizons', np.array([]))
        fh_list = list(map(int, fh_np)) if isinstance(fh_np, np.ndarray) and fh_np.size > 0 else []
        default_rca_lh = fh_list[0] if fh_list else DEFAULT_RCA_FAILURE_LOOKAHEAD
        RCA_LOOKAHEAD = int(preprocessor_data.get('rca_failure_lookahead', default_rca_lh))
        print(
            f"Using SEQ_LEN={SEQ_LEN}, RCA_FAILURE_LOOKAHEAD={RCA_LOOKAHEAD}, Found {len(canonical_sensor_names)} sensors.")
    except FileNotFoundError:
        print(f"ERROR: Preprocessor file '{PREPROCESSOR_LOAD_PATH}' not found. Exiting."); return
    except KeyError as e:
        print(f"ERROR: Missing key {e} in preprocessor file. Exiting."); return
    if RCA_LOOKAHEAD <= 0: print("ERROR: RCA_FAILURE_LOOKAHEAD must be > 0. Exiting."); return

    all_X_global_list, all_Y_per_sensor_rca_list = [], []
    file_paths = sorted(glob.glob(os.path.join(VALID_DIR, "*.csv")))
    if not file_paths: print(f"ERROR: No CSV files found in {VALID_DIR}. Exiting."); return

    print(f"\nStep 1: Preparing training data for RCA from all {len(file_paths)} CSV files...")
    for fp_idx, filepath in enumerate(file_paths):
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"    Warning: Could not read file {filepath}: {e}. Skipping."); continue
        X_f, Y_rca_f, _, _ = create_rca_input_output_windows(df, SEQ_LEN, RCA_LOOKAHEAD, canonical_sensor_names,
                                                             global_means, global_stds, os.path.basename(filepath))
        if X_f.size > 0 and Y_rca_f.size > 0 and X_f.shape[0] == Y_rca_f.shape[0]:
            all_X_global_list.append(X_f);
            all_Y_per_sensor_rca_list.append(Y_rca_f)
        if (fp_idx + 1) % 20 == 0 or (fp_idx + 1) == len(file_paths): print(
            f"  Processed {fp_idx + 1}/{len(file_paths)} files for data prep.")

    if not all_X_global_list or not all_Y_per_sensor_rca_list: print(
        "ERROR: No training data could be prepared. Exiting."); return
    X_global_train = np.concatenate(all_X_global_list, axis=0)
    Y_per_sensor_rca_train_all = np.concatenate(all_Y_per_sensor_rca_list, axis=0)

    if X_global_train.shape[0] < MIN_SAMPLES_FOR_RCA_TRAINING: print(
        f"ERROR: Insufficient samples ({X_global_train.shape[0]}). Min: {MIN_SAMPLES_FOR_RCA_TRAINING}. Exiting."); return
    if Y_per_sensor_rca_train_all.shape[0] != X_global_train.shape[0] or Y_per_sensor_rca_train_all.shape[1] != len(
        canonical_sensor_names): print(
        f"ERROR: Mismatch X/Y_rca shapes. X:{X_global_train.shape}, Y_rca:{Y_per_sensor_rca_train_all.shape}. Exiting."); return

    print(f"\nStep 2: Training one RCA classifier per sensor...")
    trained_rca_sensor_models = {}
    for s_idx, sensor_name in enumerate(canonical_sensor_names):
        Y_rca_for_this_sensor = Y_per_sensor_rca_train_all[:, s_idx]
        n_pos = np.sum(Y_rca_for_this_sensor == 1)
        n_neg = X_global_train.shape[0] - n_pos
        if X_global_train.shape[
            0] >= MIN_SAMPLES_FOR_RCA_TRAINING and n_pos >= MIN_POSITIVE_SAMPLES_FOR_RCA_TRAINING and n_neg >= MIN_POSITIVE_SAMPLES_FOR_RCA_TRAINING:
            print(
                f"  Training RCA model for: {sensor_name} ({s_idx + 1}/{len(canonical_sensor_names)}) with {X_global_train.shape[0]} samples ({n_pos} positives).")
            model = HistGradientBoostingClassifier(**HGBCLASS_PARAMS_RCA)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore"); model.fit(X_global_train, Y_rca_for_this_sensor)
                trained_rca_sensor_models[sensor_name] = model
            except Exception as e:
                print(f"    ERROR training RCA model for {sensor_name}: {e}")
        else:
            print(
                f"  Skipping RCA model for {sensor_name}: Insufficient samples or class imbalance (Total:{X_global_train.shape[0]}, Pos:{n_pos}).")
    if not trained_rca_sensor_models: print("ERROR: No RCA models trained. Exiting."); return
    print(f"  Trained RCA models for {len(trained_rca_sensor_models)} sensors.")

    print("\nStep 3: Generating classical RCA predictions (Optimized)...")
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
        if X_file_pred.shape[0] == 0: continue
        num_windows_in_file = X_file_pred.shape[0]
        num_total_sensors = len(canonical_sensor_names)

        file_all_pred_labels = np.zeros((num_windows_in_file, num_total_sensors), dtype=int)
        file_all_pred_probas = np.zeros((num_windows_in_file, num_total_sensors), dtype=float)

        for s_idx, sensor_name in enumerate(canonical_sensor_names):
            if sensor_name in trained_rca_sensor_models:
                sensor_rca_model = trained_rca_sensor_models[sensor_name]
                try:
                    file_all_pred_labels[:, s_idx] = sensor_rca_model.predict(X_file_pred)
                    file_all_pred_probas[:, s_idx] = sensor_rca_model.predict_proba(X_file_pred)[:, 1]
                except Exception as e:
                    print(f"Error during batch prediction for {sensor_name} on {os.path.basename(filepath)}: {e}")
                    # Predictions for this sensor will remain 0, 0.0

        for i in range(num_windows_in_file):
            window_info = window_info_file[i]
            gt_overall_failure_in_win = Y_fail_context_file[i]
            for s_idx, sensor_name in enumerate(canonical_sensor_names):
                output_data_list.append({
                    'filepath': window_info['filepath'],
                    'window_start_idx': window_info['window_start_idx'],
                    'rca_lookahead_horizon_config': RCA_LOOKAHEAD,
                    'gt_overall_failure_in_rca_window': gt_overall_failure_in_win,
                    'sensor_name': sensor_name,
                    'sensor_model_idx': s_idx,
                    'ground_truth_sensor_rca_label': int(Y_rca_file_gt[i, s_idx]),
                    'predicted_sensor_rca_label': file_all_pred_labels[i, s_idx],
                    'predicted_sensor_rca_probability': file_all_pred_probas[i, s_idx]
                })
        if (fp_idx + 1) % 20 == 0 or (fp_idx + 1) == len(file_paths):
            print(f"  Predicted RCA for {fp_idx + 1}/{len(file_paths)} files.")

    if output_data_list:
        output_df = pd.DataFrame(output_data_list)
        try:
            output_df.to_csv(OUTPUT_CSV_CLASSIC_RCA_FILENAME, index=False)
        except Exception as e:
            print(f"\nERROR writing CSV: {e}"); traceback.print_exc()
        else:
            print(
                f"\nSuccessfully wrote {len(output_df)} classical RCA data points to {OUTPUT_CSV_CLASSIC_RCA_FILENAME}")
    else:
        print("\nNo classical RCA data generated to write to CSV.")
    print("\n--- Classical Model RCA Script Finished ---")


if __name__ == '__main__':
    if not os.path.exists(PREPROCESSOR_LOAD_PATH): print(
        f"CRITICAL ERROR: Preprocessor file '{PREPROCESSOR_LOAD_PATH}' not found. Exiting."); exit()
    if not os.path.exists(VALID_DIR) or not os.listdir(VALID_DIR): print(
        f"CRITICAL ERROR: Validation dir '{VALID_DIR}' empty/not found. Exiting."); exit()
    test_classical_rca()
