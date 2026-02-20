import os
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm
import tempfile

# SWAP_SAVE_DIR = "swapped_data"
# os.makedirs(SWAP_SAVE_DIR, exist_ok=True)

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

def process_directory(input_dir, output_dir, plot_dir):
    """
    Process all CSV files in input_dir and save results to output_dir with plots in plot_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
   
    # Get all CSV files in input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
   
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
   
    print(f"Found {len(csv_files)} CSV files to process")
   
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        base_name = Path(csv_file).stem
        output_path = os.path.join(output_dir, f"{base_name}.csv")
       
        print(f"\nProcessing {csv_file}...")
        try:
            convert_mocap_to_qpos(input_path, output_path, plot_dir, base_name)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert mocap data to qpos arrays")
    parser.add_argument("--input_dir", required=True, help="Directory containing input mocap CSV files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output qpos CSV files")
    parser.add_argument("--plot_dir", required=True, help="Directory to save joint angle plots")
   
    args = parser.parse_args()
   
    process_directory(args.input_dir, args.output_dir, args.plot_dir)