"""
This script takes in 2 directories: mocap and sensor. It then edits the sensor data to make the time continuous and resamples the mocap data to fit the same number of columns as the sensor data. 
The sensor data is then normalized based on the threshold by dividing the capacitances by the average of the baseline pose capacitances.
Additionally, it calculates the angle arrays and saves the angle arrays into a folder as well as its respective plots.  

"""

import os 
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm

# SENSOR_NAMES = [
#     "1. right elbow", "2. right shoulder", "3. right collarbone", "4. bottom back", "5. top back",
#     "6. left collarbone", "7. left shoulder", "8. left armpit (back)", "9. left elbow", "10. left elbow (back)",
#     "11. left armpit (front)", "12. waist left a", "13. chest l", "14. stomach l", "15. hip right",
#     "16. waist left b", "17. right armpit (back)", "18. right elbow (front)", "19. right armpit (front)",
#     "20. waist right a", "21. waist right b", "22. hip left", "23. stomach r", "24. chest r"
# ]

def fix_time_wraparound(time_series): 
    """
    Fix wraparound in time series data by adjusting negative deltas.
    This function assumes time values are in milliseconds and wraps around at 65536 ms.
    looks correct
    """
    corrected_time = [time_series.iloc[0]]
    for i in range(1, len(time_series)):
        delta = time_series.iloc[i] - time_series.iloc[i - 1]
        if delta < 0:
            delta += 65536
        corrected_time.append(corrected_time[-1] + delta)
    result = pd.Series(corrected_time)
    # print(result)
    return result

def process_sensor_file(input_file): 
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
        print("ori", df.head())
        if df.shape[0] < 2:
            raise ValueError(f"Input file {input_file} has fewer than 2 rows.")
        df = df.iloc[1:].reset_index(drop=True)
        if 'Time_ms' not in df.columns:
            raise ValueError(f"Input file {input_file} does not contain 'Time_ms' column.")
        
        # if 'Sensor_7' in df.columns:
        #     print("Dropping column: Sensor_7")
        #     df = df.drop(columns=['Sensor_7'])
        
        # df = df[(df <= 1000).all(axis=1)]
        sensor_columns = [col for col in df.columns if col != 'Time_ms']
        df = df[(df[sensor_columns] <= 500).all(axis=1)]

        df['Time_ms'] = fix_time_wraparound(df['Time_ms'])

        df['Time_ms'] -= df['Time_ms'].iloc[0]
        # print("new", df.head())
        return df
    else:
        raise ValueError(f"Input file {input_file} is not a CSV file.")

def normalize_sensor_by_baseline(sensor_df, baseline_duration_ms=5000, method="relative"):
    """
    Normalize sensor data based on baseline duration.
    Baseline is defined as the first 'baseline_duration_ms' milliseconds of data.
    """
    time_col = sensor_df.iloc[:, 0]
    sensor_data = sensor_df.iloc[:, 1:]

    baseline_mask = time_col <= baseline_duration_ms
    baseline_rows = sensor_data[baseline_mask]

    baseline_means = baseline_rows.mean()
    # print(f"Baseline means:\n{baseline_means}")

    normalized = sensor_df.copy()

    if method == "relative":
        normalized.iloc[:, 1:] = sensor_data / baseline_means.values
    elif method == "delta":
        normalized.iloc[:, 1:] = sensor_data - baseline_means.values
    else:
        raise ValueError("Normalization method must be 'relative' or 'delta'")

    return normalized

def find_timestamp_column(df_mocap): 
    for col in df_mocap.columns: 
        lower = col.lower()
        if lower == "time_ms": 
            if col != "time_ms":
                df_mocap.rename(columns={col: 'time_ms'}, inplace=True)
                # print(f"Renamed {col} to 'time_ms'")
            else: 
                print("Found the time_ms column")
    return df_mocap.columns[0]

def resample_to_sensor(df_mocap, df_sensor, time_col): 
    """
    Resample mocap data to match sensor timestamps.
    """
    mocap_times = df_mocap[time_col].values
    sensor_times = df_sensor[time_col].values

    sort_idx = np.argsort(mocap_times)
    mocap_times = mocap_times[sort_idx]
    mocap_sorted = df_mocap.iloc[sort_idx].reset_index(drop=True)

    positions = np.searchsorted(mocap_times, sensor_times)
    pos_clip = np.clip(positions, 1, len(mocap_times) - 1)

    left_times = mocap_times[pos_clip - 1]
    right_times = mocap_times[pos_clip]
    choose_left = np.abs(sensor_times - left_times) <= np.abs(sensor_times - right_times)

    chosen_idx = pos_clip.copy()
    chosen_idx[choose_left] = pos_clip[choose_left] - 1

    resampled = mocap_sorted.iloc[chosen_idx].reset_index(drop=True)

    resampled[time_col] = df_sensor[time_col].values
    return resampled

def plot_sensors(csv_path, output_path): 
    df = pd.read_csv(csv_path)

    if df.shape[1] != 25:
        print(f"Skipping {csv_path}: Expected 25 columns (1 time + 24 sensors), got {df.shape[1]}")
        return

    time = df.iloc[:, 0].to_numpy()
    sensor_cols = df.columns[1:]

    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 15), sharex=True)
    fig.suptitle(f"Sensors: {os.path.basename(csv_path)}", fontsize=18)

    for idx, col in enumerate(sensor_cols):
        row = idx // 4
        col_idx = idx % 4
        ax = axes[row, col_idx]
        ax.plot(time, df[col].to_numpy(), linewidth=1.0)
        ax.set_title(f"Sensor {idx + 1}", fontsize=9)  # Generic title based on index
        ax.set_ylabel("Value")
        if row == 5:
            ax.set_xlabel("Time (ms)")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def process_mocap_and_sensor_data(mocap_dir, sensor_dir, output_dir):
    """
    Process mocap and sensor data to normalize sensor readings based on mocap transition points.
    """
    # Ensure Path objects
    mocap_dir = Path(mocap_dir)
    sensor_dir = Path(sensor_dir)
    output_dir = Path(output_dir)

    print(f"Given the following paths: {mocap_dir}, {sensor_dir}, {output_dir}")

    sensor_list = list(sensor_dir.glob('*_CapacitanceTable.csv'))
    # print(f"DEBUG: sensor_dir = {sensor_dir}")
    # print(f"DEBUG: found {len(sensor_list)} sensor files:", sensor_list)

    output_dir.mkdir(parents=True, exist_ok=True)

    for sensor_file in sensor_list:
        base_name = sensor_file.stem.replace('_CapacitanceTable', '')
        mocap_file = mocap_dir / f"{base_name}.csv"

        if not mocap_file.exists():
            print(f"Warning: No matching mocap file for {sensor_file.name}")
            continue
        
        # print(f"Found matches for sensor: {sensor_file} and mocap: {mocap_file}")
        sensor_df = pd.read_csv(sensor_file)
        mocap_df  = pd.read_csv(mocap_file)

        ts_col = find_timestamp_column(sensor_df)

        resampled_df = resample_to_sensor(mocap_df, sensor_df, ts_col)

        keep_cols = ['time_ms', 'base_X.1', 'base_Y.1', 'base_Z.1', 'base_X', 'base_Y', 'base_Z', 'base_W',
                     'abdomen_X', 'abdomen_Y', 'abdomen_Z', 'abdomen_W', 'left_thigh_X', 'left_thigh_Y',
                     'left_thigh_Z', 'left_thigh_W', 'right_thigh_X', 'right_thigh_Y', 'right_thigh_Z',
                     'right_thigh_W', 'left_upper_arm_X', 'left_upper_arm_Y', 'left_upper_arm_Z',
                     'left_upper_arm_W', 'right_upper_arm_X', 'right_upper_arm_Y', 'right_upper_arm_Z',
                     'right_upper_arm_W', 'left_forearm_X', 'left_forearm_Y', 'left_forearm_Z',
                     'left_forearm_W', 'right_forearm_X', 'right_forearm_Y', 'right_forearm_Z',
                     'right_forearm_W']

        resampled_df = resampled_df[keep_cols]

        output_path = output_dir / f"{base_name}_resamp.csv"
        resampled_df.to_csv(output_path, index=False)
        # print(f"Resampled data saved to: {output_path}")

def safe_rotation(w, x, y, z, eps=1e-8):
    """Build a scipy Rotation from quaternion (w,x,y,z) safely."""
    q = np.array([x, y, z, w], dtype=float)
    if np.any(np.isnan(q)) or np.linalg.norm(q) < eps:
        return R.identity()
    return R.from_quat(q / np.linalg.norm(q))

def compute_hinge_angle(parent_mat, child_mat, axis):
    """Signed angle about `axis` between parent→child frames."""
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    axis_world = child_mat @ axis
    rel = parent_mat.T @ child_mat
    rot = R.from_matrix(rel)
    rotvec_parent = rot.as_rotvec()
    rotvec_world = parent_mat @ rotvec_parent
    return float(rotvec_world.dot(axis_world))

def load_clean_data(filename):
    df = pd.read_csv(filename, header=0, skipinitialspace=True)
    df.columns = [c.strip().replace('"', '') for c in df.columns]
    return df.dropna(how='all')

def plot_joint_angles(df, joints, output_dir, pic_name):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    colors = cm.tab20(np.linspace(0, 1, len(joints)))
    # t = df['time_ms'].values if 'time_ms' in df.columns else np.arange(len(df))
    t = np.arange(len(df))
    for name, color in zip(joints, colors):
        plt.plot(t, df[name].values, label=name, color=color)
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (rad)')
    plt.title('Revolute Joint Angles')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{pic_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")

def convert_mocap_to_qpos(input_csv, output_csv, plot_dir, pic_name):
    df = load_clean_data(input_csv)

    # COMMENT OUT LATER!!!
    swap_pairs = [
        ('_W', '_Y'),
        ('_X', '_Z')
    ]


    for suffix_a, suffix_b in swap_pairs:
        cols_a = [c for c in df.columns if c.endswith(suffix_a)]
        for col_a in cols_a:
            col_b = col_a[:-2] + suffix_b
            if col_b in df.columns:
                tmp = df[col_a].copy()
                df[col_a] = df[col_b]
                df[col_b] = tmp
    
    # negative X and Y
    for col in df.columns:
        if col.endswith('_X') or col.endswith('_Y'):
            df[col] = -df[col]
    
    # save the df in a temporary folder
    # save_path = os.path.join(SWAP_SAVE_DIR, f"{pic_name}_swapped.csv")
    # df.to_csv(save_path, index=False)
    # print(f"Saved swapped DataFrame to: {save_path}")


    # if 'base_W' in df.columns:
    #     df['base_W'] = df['base_W'] - 1

    # if 'base_Y' in df.columns:
    #     df['base_Y'] = df['base_Y'] + 1

    n  = len(df)

    # ---- Detect available base columns ----
    base_pos_cols  = ['base_X.1', 'base_Y.1', 'base_Z.1']
    base_quat_cols = ['base_W', 'base_X', 'base_Y', 'base_Z']
    has_base_pos   = all(c in df.columns for c in base_pos_cols)
    has_base_quat  = all(c in df.columns for c in base_quat_cols)

    if not has_base_pos:
        print("[INFO] No base position columns found (base_X.1/base_Y.1/base_Z.1). "
              "Skipping x,y,z in output.")
    if not has_base_quat:
        print("[INFO] No base quaternion columns found (base_W/base_X/base_Y/base_Z). "
              "Skipping qw,qx,qy,qz in output. Base frame = identity for angle calc.")

    # Positions: mm → m (if base pos present)
    if has_base_pos:
        pos_new = np.vstack((df['base_X.1'], df['base_Y.1'], df['base_Z.1'])).T / 1000.0

    joints = [
        'left_hip_roll_joint', 'right_hip_roll_joint',
        'waist_yaw_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
        'waist_pitch_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint',
        'waist_roll_joint',
        'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
        'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
        'left_elbow_pitch_joint', 'right_elbow_pitch_joint'
    ]

    joints2 = ['qx', 'qy', 'qz', 'qw',
        'left_hip_roll_joint', 'right_hip_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint',
        'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
        'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
        'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
        'left_elbow_pitch_joint', 'right_elbow_pitch_joint'
    ]

    # ---- Build output dict dynamically (skip base pose if absent) ----
    out = {}
    if 'time_ms' in df.columns:
        out['time_ms'] = df['time_ms'].values
    else:
        print("[WARN] 'time_ms' not found; writing synthetic time index.")
        out['time_ms'] = np.arange(n)

    if has_base_pos:
        # Note the coordinate remap used before: x<-Z, y<-X, z<-Y
        out['x'] = pos_new[:, 2]
        out['y'] = pos_new[:, 0]
        out['z'] = pos_new[:, 1]

    if has_base_quat:
        out['qw'] = np.zeros(n)
        out['qx'] = np.zeros(n)
        out['qy'] = np.zeros(n)
        out['qz'] = np.zeros(n)

    for j in joints:
        out[j] = np.zeros(n)

    # Helper: safe seg rot; if any seg quaternion columns missing, return identity
    def seg_mat(seg, idx):
        need = [f'{seg}_W', f'{seg}_X', f'{seg}_Y', f'{seg}_Z']
        if not all(c in df.columns for c in need):
            return np.eye(3)
        r = safe_rotation(
            df.loc[idx, f'{seg}_W'],
            df.loc[idx, f'{seg}_X'],
            df.loc[idx, f'{seg}_Y'],
            df.loc[idx, f'{seg}_Z']
        )
        # Swap (basis mapping) as in original
        P_yz = np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]])
        return P_yz @ r.as_matrix()

    for i in range(n):
        # Base quaternion (if present) else identity
        if has_base_quat:
            r_base = safe_rotation(
                df.loc[i, 'base_W'],
                df.loc[i, 'base_X'],
                df.loc[i, 'base_Y'],
                df.loc[i, 'base_Z']
            )
            qx, qy, qz, qw = r_base.as_quat()
            out['qw'][i], out['qx'][i], out['qy'][i], out['qz'][i] = qw, qz, qx, qy
            # out['qw'][i], out['qx'][i], out['qy'][i], out['qz'][i] = qw, qx, qy, qz

            R_base = seg_mat('base', i)
            # print(R_base)
        else:
            r_base = R.identity()
            R_base = np.eye(3)

        R_abd  = seg_mat('abdomen', i)
        R_lth  = seg_mat('left_thigh', i)
        R_rth  = seg_mat('right_thigh', i)
        R_lsh  = seg_mat('left_upper_arm', i)
        R_rsh  = seg_mat('right_upper_arm', i)
        R_lfa  = seg_mat('left_forearm', i)
        R_rfa  = seg_mat('right_forearm', i)

        # yaw, pitch, roll = r_base.as_euler('ZYX')
        # print(f"yaw: {yaw}, quat_z: {df.loc[i, 'base_Z']}")
        # print(yaw == out[qz][i])
        # print(yaw, pitch, roll)

        # Hip angles relative to base (or world if base missing)
        out['left_hip_roll_joint'][i]   = compute_hinge_angle(R_base, R_lth, [1,0,0])
        out['right_hip_roll_joint'][i]  = compute_hinge_angle(R_base, R_rth, [1,0,0])
        out['left_hip_yaw_joint'][i]    = compute_hinge_angle(R_base, R_lth, [0,0,1])
        out['right_hip_yaw_joint'][i]   = compute_hinge_angle(R_base, R_rth, [0,0,1])
        out['left_hip_pitch_joint'][i]  = compute_hinge_angle(R_base, R_lth, [0,1,0])
        out['right_hip_pitch_joint'][i] = compute_hinge_angle(R_base, R_rth, [0,1,0])

        # Waist relative to base (or world if base missing)
        out['waist_yaw_joint'][i]       = out['qz'][i] if has_base_quat else 0.0
        out['waist_pitch_joint'][i]     = out['qy'][i] if has_base_quat else 0.0
        out['waist_roll_joint'][i]      = out['qx'][i] if has_base_quat else 0.0

        # out['waist_yaw_joint'][i]       = compute_hinge_angle(R_base, R_abd, [0,0,1])
        # out['waist_pitch_joint'][i]     = compute_hinge_angle(R_base, R_abd, [0,1,0])
        # out['waist_roll_joint'][i]      = compute_hinge_angle(R_base, R_abd, [1,0,0])

        # Shoulders relative to abdomen
        out['left_shoulder_pitch_joint'][i]  = compute_hinge_angle(R_abd, R_lsh, [0,1,0])
        out['right_shoulder_pitch_joint'][i] = compute_hinge_angle(R_abd, R_rsh, [0,-1,0])
        out['left_shoulder_roll_joint'][i]   = compute_hinge_angle(R_abd, R_lsh, [1,0,0])
        out['right_shoulder_roll_joint'][i]  = compute_hinge_angle(R_abd, R_rsh, [-1,0,0])
        out['left_shoulder_yaw_joint'][i]    = compute_hinge_angle(R_abd, R_lsh, [0,0,1])
        out['right_shoulder_yaw_joint'][i]   = compute_hinge_angle(R_abd, R_rsh, [0,0,1])

        # Elbows relative to upper arms
        out['left_elbow_pitch_joint'][i]     = compute_hinge_angle(R_lsh, R_lfa, [0,1,0])
        out['right_elbow_pitch_joint'][i]    = compute_hinge_angle(R_rsh, R_rfa, [0,-1,0])

    df_out = pd.DataFrame(out)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved qpos array ({len(joints)} joints{'' if has_base_quat else ', no base pose columns'}) to {output_csv}")

    plot_joint_angles(df_out, joints, plot_dir, pic_name)
    return df_out

if __name__ == "__main__": 
    p = argparse.ArgumentParser(description="Fix sensor Time_ms wraparound and drop first row")
    p.add_argument("--sensor_dir", required=True, help="Path to input sensor CSV folder")
    p.add_argument("--mocap_dir", required=True, help="Path to write cleaned CSV folder")
    p.add_argument("--sensor_plots", required=True, help="Path to write sensor plots")
    p.add_argument("--sensor_output", required=True, help="Path to write cleaned sensor CSV file")
    p.add_argument("--mocap_output", required=True, help="Path to write cleaned mocap CSV file")
    p.add_argument("--angle_arrays", required=True, help="Path to write cleaned angle array CSV file")
    p.add_argument("--mocap_plots", required=True, help="Path to write mocap plots")

    args = p.parse_args()
    angle_dir = Path(args.angle_arrays)
    sensor_plots_dir = args.sensor_plots
    sensor_dir = args.sensor_dir
    mocap_dir = args.mocap_dir
    sensor_output_dir = args.sensor_output
    mocap_output_dir = Path(args.mocap_output)
    mocap_plots_dir = args.mocap_plots
    if not os.path.exists(sensor_output_dir):
        os.makedirs(sensor_output_dir)
    if not os.path.exists(mocap_output_dir):
        os.makedirs(mocap_output_dir)
    for filename in os.listdir(sensor_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(sensor_dir, filename)
            try:
                sensor_df = process_sensor_file(input_path)
                # print(sensor_df)
                # print(sensor_df.head())
                # Normalize sensor data
                # sensor_df = normalize_sensor_by_baseline(sensor_df, baseline_duration_ms=5000, method="relative")

                output_path = os.path.join(sensor_output_dir, filename)
                sensor_df.to_csv(output_path, index=False)
                # print(f"Processed and saved {filename} to {output_path}")
                # Plot sensors
                plot_output_path = os.path.join(sensor_plots_dir, f"{filename[:-4]}_sensors.png")
                plot_sensors(output_path, plot_output_path)
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except ValueError as e:
                print(f"Value error in {filename}: {e}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    # Process mocap data
    process_mocap_and_sensor_data(mocap_dir, sensor_output_dir, mocap_output_dir)
    for csv_file in sorted(mocap_output_dir.glob("*.csv")):
        base = csv_file.stem
        input_csv  = csv_file
        output_csv = angle_dir / f"{base}.csv"          # <— save here
        convert_mocap_to_qpos(input_csv, output_csv, mocap_plots_dir, base)

    print("\nProcessing complete!")