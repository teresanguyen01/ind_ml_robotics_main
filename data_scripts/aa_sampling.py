import os
import random 
import numpy as np
import pandas as pd

def detect_peaks(signal, threshold=None):
    signal = np.asarray(signal)
    if threshold is None: 
        threshold = np.mean(signal) + 0.1 * np.std(signal)
    
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)
    return np.array(peaks)

def extract_cycles_from_dataframe(df, cycle_joint_name, min_cycle_length=10):
    if cycle_joint_name not in df.columns:
        raise ValueError(f"Joint name '{cycle_joint_name}' not found in DataFrame columns.")
    
    signal = df[cycle_joint_name].to_numpy()
    signal = signal - np.mean(signal)

    peak_indicies = detect_peaks(signal)
    if len(peak_indicies) < 2:
        raise ValueError("Not enough peaks detected to extract cycles.")
    cycles = []

    arr = df.to_numpy()
    for i in range(len(peak_indicies) - 1):
        start_idx = peak_indicies[i]
        end_idx = peak_indicies[i + 1]
        if end_idx - start_idx >= min_cycle_length:
            cycle = arr[start_idx:end_idx, :]
            cycles.append(cycle)
    return cycles

def collect_all_cycles(input_dir, cycle_joint_name, min_cycle_length=10, file_ext=".csv"):
    all_cycles = []
    for filename in os.listdir(input_dir):
        if filename.endswith(file_ext):
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path)
            try:
                cycles = extract_cycles_from_dataframe(df, cycle_joint_name, min_cycle_length)
                all_cycles.extend(cycles)
                print(f"Extracted {len(cycles)} cycles from {filename}")
            except ValueError as e:
                print(f"Skipping {filename}: {e}")
    return all_cycles

def create_mixed_cycle_sequence(input_dir, output_path, num_output_cycles=20, cycle_joint_name="right_elbow", min_cycle_length=10, file_ext=".csv"):
    sample_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(file_ext)]
    if not sample_files:
        raise ValueError(f"No files found in {input_dir} with extension {file_ext}")
    
    sample_df = pd.read_csv(sample_files[0])
    columns = list(sample_df.columns)

    all_cycles = collect_all_cycles(input_dir, cycle_joint_name, min_cycle_length, file_ext)
    chosen_cycles = []
    for _ in range(num_output_cycles):
        cycle = random.choice(all_cycles)
        chosen_cycles.append(cycle)
    
    new_sequence = np.concatenate(chosen_cycles, axis=0)

    new_df = pd.DataFrame(new_sequence, columns=columns)
    new_df.to_csv(output_path, index=False)
    print(f"Saved mixed cycle sequence to {output_path}")
#         print(f"Error processing {csv_file}: {e}")
    return new_df

if __name__ == "__main__":
    """
    qw,qx,qy,qz,left_hip_roll_joint,right_hip_roll_joint,waist_yaw_joint,left_hip_yaw_joint,
    right_hip_yaw_joint,waist_pitch_joint,left_hip_pitch_joint,right_hip_pitch_joint,
    waist_roll_joint,left_shoulder_pitch_joint,right_shoulder_pitch_joint,left_shoulder_roll_joint,
    right_shoulder_roll_joint,left_shoulder_yaw_joint,right_shoulder_yaw_joint,left_elbow_pitch_joint,
    right_elbow_pitch_joint
    """
    INPUT_DIR = "AA_SENSOR-RAWAN-VERO-TERE/aa_train"
    OUTPUT_DIR = "test.csv"

    CYCLE_JOINT_NAME = "right_elbow_pitch_joint"
    MIN_CYCLE_LENGTH = 15
    NUM_OUTPUT_CYCLES = 30

    create_mixed_cycle_sequence(INPUT_DIR, OUTPUT_DIR, NUM_OUTPUT_CYCLES, CYCLE_JOINT_NAME, MIN_CYCLE_LENGTH)