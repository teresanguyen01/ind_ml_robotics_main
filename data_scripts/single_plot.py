import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_joint_angles(df, joints, output_dir, pic_name):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    colors = cm.tab20(np.linspace(0, 1, len(joints)))
    
    t = np.arange(len(df))
    for name, color in zip(joints, colors):
        if name not in df.columns:
            print(f"[WARN] Column {name} not found in {pic_name}, skipping.")
            continue
        plt.plot(t, df[name].values, label=name, color=color)

    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (rad)')
    plt.title(f'Revolute Joint Angles – {pic_name}')
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'{pic_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_path}")


def process_directory(input_dir, output_dir, joints):
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for fname in csv_files:
        fpath = os.path.join(input_dir, fname)
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"[ERROR] Failed to read {fname}: {e}")
            continue

        pic_name = os.path.splitext(fname)[0]
        plot_joint_angles(df, joints, output_dir, pic_name)


# ------------------------------
# Choose your joints here
# ------------------------------
joints = ['left_upper_arm_X', 'left_upper_arm_Y', 'left_upper_arm_Z', 'left_upper_arm_W', 'right_upper_arm_X', 'right_upper_arm_Y', 'right_upper_arm_Z', 'right_upper_arm_W']

# ------------------------------
# Run on entire directory
# ------------------------------
INPUT_DIR = "AA-MAIN-FOLDER/Yan_all_data/YAN_0120_cleaned_mocap"              # <--- your folder containing many CSVs
OUTPUT_DIR = "AA-MAIN-FOLDER/Yan_all_data/mocap_plots"  # <--- one folder for all plots

process_directory(INPUT_DIR, OUTPUT_DIR, joints)
