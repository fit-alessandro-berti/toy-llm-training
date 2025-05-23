import os
import uuid
import random
import numpy as np
import pandas as pd

# --- Configuration Constants ---
TOTAL_TYPES = 35
TRAIN_VAL_TYPES_END = 30  # Types 1-30 for Training/Validation
FILES_PER_TYPE_CONFIG = 10  # Number of CSV files to generate for each type configuration
TRAIN_RATIO = 0.8

# Folders
BASE_DIR = "."
TRAIN_FOLDER = os.path.join(BASE_DIR, "TRAINING")
VALIDATION_FOLDER = os.path.join(BASE_DIR, "VALIDATION")
TEST_FOLDER = os.path.join(BASE_DIR, "TEST")

# --- Helper Functions ---

def create_dirs():
    """Creates the necessary directories for saving the data."""
    os.makedirs(TRAIN_FOLDER, exist_ok=True)
    os.makedirs(VALIDATION_FOLDER, exist_ok=True)
    os.makedirs(TEST_FOLDER, exist_ok=True)
    print(f"Directories created/ensured: {TRAIN_FOLDER}, {VALIDATION_FOLDER}, {TEST_FOLDER}")

def generate_type_config(type_id):
    """
    Generates a unique configuration dictionary for a given time series type.
    type_id is 1 to 35.
    """
    config = {}
    config['type_id'] = type_id

    # Introduce variability based on type_id; e.g., divide types into groups
    group = (type_id - 1) // 7  # Results in 5 groups (0 to 4) for 35 types

    # Number of sensors: increases with group, e.g., group 0 (types 1-7) might have 3-7 sensors,
    # group 4 (types 29-35) might have 7-15 sensors.
    config['num_sensors'] = random.randint(3 + group, 7 + group * 2)
    # Series length: also varies with group.
    config['series_length'] = random.randint(600 + group * 200, 1200 + group * 400)

    # Sensor-specific parameters
    sensor_params_list = []
    for i in range(config['num_sensors']):
        sensor_p = {
            'base_mean': random.uniform(-5 * (group + 0.5), 5 * (group + 0.5)),
            'base_std': random.uniform(0.5 + group * 0.1, 1.5 + group * 0.3),
            # Trend: Some sensors have trend, some don't. Slope varies.
            'trend_slope': random.uniform(-0.003 * (group + 1), 0.003 * (group + 1)) * random.choice([0,0,1]),
            # Seasonality: Likelihood and strength increase slightly with group.
            'has_seasonality': random.random() < (0.3 + group * 0.1),
        }
        if sensor_p['has_seasonality']:
            sensor_p['seasonality_amplitude'] = random.uniform(1 + group * 0.5, 3 + group * 1.5)
            sensor_p['seasonality_period'] = random.choice([24, 40, 60, 80, 120])
        else:
            sensor_p['seasonality_amplitude'] = 0
            sensor_p['seasonality_period'] = 0
        # AR(1) component for noise: Some sensors have it.
        sensor_p['ar_phi'] = random.uniform(0.2, 0.7) if random.random() < (0.4 + group*0.1) else 0
        sensor_params_list.append(sensor_p)
    config['sensor_params'] = sensor_params_list

    # Anomaly and Failure parameters
    config['subtle_anomaly_rate'] = random.uniform(0.01, 0.04) # Proportion of data points (t,s) that are subtly anomalous
    config['subtle_anomaly_magnitude_factor'] = random.uniform(1.5, 3.0)  # Multiplier for sensor_std

    config['num_major_failures'] = random.randint(1, 2 + group // 2) # e.g., 1-2 for group 0, up to 1-4 for group 4
    
    # Ensure trigger duration is reasonable for series length
    min_trig_dur = max(10, config['series_length'] // 100)
    max_trig_dur = max(20, config['series_length'] // 25)
    config['trigger_duration'] = random.randint(min_trig_dur, max_trig_dur)
    config['failure_duration'] = random.randint(1, 3 + group // 2)

    # Ratio of sensors affected by triggers/failures
    config['trigger_sensors_ratio'] = random.uniform(0.2 + group * 0.04, 0.4 + group * 0.08)
    config['failure_sensors_ratio'] = random.uniform(0.3 + group * 0.05, 0.6 + group * 0.1)

    # Magnitude of change during triggers/failures (multiplier for sensor_std)
    config['trigger_value_change_factor'] = random.uniform(1.0 + group * 0.1, 2.0 + group * 0.2)
    config['failure_value_change_factor'] = random.uniform(3.5 + group * 0.3, 6.0 + group * 0.5)
    
    # Make later types potentially more complex or noisy
    if type_id > 28: # Last group of types (29-35)
        config['subtle_anomaly_rate'] = min(0.06, config['subtle_anomaly_rate'] * 1.5) # More subtle anomalies
        config['num_major_failures'] = max(config['num_major_failures'], 2 + group //2) # Ensure at least a couple failures

    return config


def generate_one_timeseries_instance(config):
    """Generates a single multivariate time series DataFrame based on the given config."""
    num_sensors = config['num_sensors']
    series_length = config['series_length']
    sensor_configs = config['sensor_params']

    data = np.zeros((series_length, num_sensors))
    time_vector = np.arange(series_length)
    sensor_actual_stds = []

    # 1. Generate base data for each sensor
    for i in range(num_sensors):
        s_config = sensor_configs[i]
        signal = np.full(series_length, s_config['base_mean'])

        # Trend
        if s_config.get('trend_slope', 0) != 0:
            signal += s_config['trend_slope'] * time_vector

        # Seasonality
        if s_config.get('has_seasonality', False) and s_config.get('seasonality_period', 0) > 0:
            signal += s_config['seasonality_amplitude'] * np.sin(2 * np.pi * time_vector / s_config['seasonality_period'])

        # Noise (potentially AR(1))
        base_noise = np.random.normal(0, s_config['base_std'], series_length)
        if s_config.get('ar_phi', 0) != 0 and abs(s_config['ar_phi']) < 1:
            ar_noise = np.zeros(series_length)
            ar_noise[0] = base_noise[0]
            for t in range(1, series_length):
                ar_noise[t] = s_config['ar_phi'] * ar_noise[t-1] + base_noise[t] # (1-phi^2) for variance preservation if needed
            signal += ar_noise
        else:
            signal += base_noise
        
        data[:, i] = signal
        # Store the std of the generated base signal for scaling anomalies/failures meaningfully
        actual_std = np.std(signal)
        sensor_actual_stds.append(max(actual_std, 0.1)) # Avoid zero std for flat signals

    current_failure_flags = np.zeros(series_length, dtype=int)

    # 2. Introduce subtle anomalies (CURRENT_FAILURE = 0)
    num_subtle_anomalies_total = int(config['subtle_anomaly_rate'] * series_length * num_sensors)
    for _ in range(num_subtle_anomalies_total):
        t_idx = random.randint(0, series_length - 1)
        s_idx = random.randint(0, num_sensors - 1)
        
        # Ensure this point is not already part of a major failure (important if failures are injected first)
        # For now, assume major failures might overwrite these, or place them carefully.
        # if current_failure_flags[t_idx] == 1: continue # Skip if already a major failure point
        
        magnitude_change = config['subtle_anomaly_magnitude_factor'] * sensor_actual_stds[s_idx]
        data[t_idx, s_idx] += random.choice([-1, 1]) * magnitude_change
        # current_failure_flags[t_idx] remains 0 for these

    # 3. Introduce major failures with triggers (CURRENT_FAILURE = 1 for failure points)
    num_failures_to_inject = config['num_major_failures']
    trigger_duration = config['trigger_duration']
    failure_duration = config['failure_duration']
    
    placed_event_intervals = [] # Stores (event_start_time, event_end_time)

    for _ in range(num_failures_to_inject):
        max_placement_attempts = 50
        for attempt in range(max_placement_attempts):
            # Potential failure_start_idx must allow for full trigger before and full failure duration
            min_start = trigger_duration
            max_start = series_length - failure_duration -1 
            
            if min_start >= max_start : # Not enough room for any event
                break 

            potential_failure_start_time = random.randint(min_start, max_start)
            
            event_actual_start = potential_failure_start_time - trigger_duration
            event_actual_end = potential_failure_start_time + failure_duration

            # Check for overlap
            is_overlapping = False
            for (s, e) in placed_event_intervals:
                if max(event_actual_start, s) < min(event_actual_end, e):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                placed_event_intervals.append((event_actual_start, event_actual_end))
                
                # --- Apply Trigger ---
                num_trigger_sens = max(1, int(num_sensors * config['trigger_sensors_ratio']))
                trigger_sensor_indices = random.sample(range(num_sensors), k=min(num_trigger_sens, num_sensors))
                
                for t in range(event_actual_start, potential_failure_start_time):
                    if t < 0 or t >= series_length : continue # Should not happen with proper start time selection
                    for s_idx in trigger_sensor_indices:
                        change = config['trigger_value_change_factor'] * sensor_actual_stds[s_idx]
                        progression = (t - event_actual_start + 1) / trigger_duration # Gradual effect
                        data[t, s_idx] += random.choice([-1,1]) * change * progression * random.uniform(0.7,1.3) # Add some randomness to change
                    # current_failure_flags[t] remains 0

                # --- Apply Failure ---
                num_failure_sens = max(1, int(num_sensors * config['failure_sensors_ratio']))
                failure_sensor_indices = random.sample(range(num_sensors), k=min(num_failure_sens, num_sensors))

                for t in range(potential_failure_start_time, event_actual_end):
                    if t < 0 or t >= series_length: continue
                    current_failure_flags[t] = 1
                    for s_idx in failure_sensor_indices:
                        f_change = config['failure_value_change_factor'] * sensor_actual_stds[s_idx]
                        # Different types of failure manifestations
                        failure_type = random.choice(["spike", "level_shift_extreme", "bias"])
                        if failure_type == "spike":
                            data[t, s_idx] += random.choice([-1, 1]) * f_change * random.uniform(0.8, 1.5)
                        elif failure_type == "level_shift_extreme": # Shift to a new, very different level
                            data[t, s_idx] = sensor_configs[s_idx]['base_mean'] + random.choice([-1,1]) * f_change
                        elif failure_type == "bias": # Add a significant bias
                            data[t, s_idx] += random.choice([-1,1]) * f_change * 0.5 # A persistent offset
                break # Successfully placed and applied this failure event
        # If max_placement_attempts reached, this failure event is skipped for this series instance

    # Create DataFrame
    sensor_cols = [f'Sensor{i+1}' for i in range(num_sensors)]
    df = pd.DataFrame(data, columns=sensor_cols)
    df['CURRENT_FAILURE'] = current_failure_flags
    return df

# --- Main Script Logic ---
def main():
    create_dirs()
    
    print(f"Generating {TOTAL_TYPES} types of time series data...")
    all_type_configs = [generate_type_config(i) for i in range(1, TOTAL_TYPES + 1)]
    
    for type_idx, current_type_config in enumerate(all_type_configs):
        num_type = current_type_config['type_id'] # This is type_id from 1 to 35
        print(f"\nProcessing Type {num_type}/{TOTAL_TYPES}...")
        print(f"  Config: {current_type_config['num_sensors']} sensors, {current_type_config['series_length']} length, "
              f"{current_type_config['num_major_failures']} major failures.")

        generated_files_for_this_type = []
        for i in range(FILES_PER_TYPE_CONFIG):
            df = generate_one_timeseries_instance(current_type_config)
            random_uuid = str(uuid.uuid4().hex)[:12] # Shorter UUID
            filename = f"{num_type}_{random_uuid}.csv"
            generated_files_for_this_type.append({'df': df, 'filename': filename})
            if (i+1) % (FILES_PER_TYPE_CONFIG // 2) == 0 : # Print progress for long generations
                 print(f"    Generated instance {i+1}/{FILES_PER_TYPE_CONFIG} for type {num_type}")


        # Distribute files
        if 1 <= num_type <= TRAIN_VAL_TYPES_END:
            split_index = int(FILES_PER_TYPE_CONFIG * TRAIN_RATIO)
            train_files = generated_files_for_this_type[:split_index]
            val_files = generated_files_for_this_type[split_index:]

            for f_info in train_files:
                filepath = os.path.join(TRAIN_FOLDER, f_info['filename'])
                f_info['df'].to_csv(filepath, index=False)
            
            for f_info in val_files:
                filepath = os.path.join(VALIDATION_FOLDER, f_info['filename'])
                f_info['df'].to_csv(filepath, index=False)
            print(f"  Type {num_type}: Saved {len(train_files)} to TRAINING, {len(val_files)} to VALIDATION.")

        else: # Test types (TRAIN_VAL_TYPES_END < num_type <= TOTAL_TYPES)
            for f_info in generated_files_for_this_type:
                filepath = os.path.join(TEST_FOLDER, f_info['filename'])
                f_info['df'].to_csv(filepath, index=False)
            print(f"  Type {num_type}: Saved {len(generated_files_for_this_type)} to TEST.")
            
    print("\nTime series generation complete!")
    print(f"Data saved in: {os.path.abspath(BASE_DIR)}")

if __name__ == '__main__':
    main()