#!/usr/bin/env python3
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import genesis as gs

# ---- Joint list (edit if your CSV has different joints) -------------------- #
DOF_NAMES = [
    'left_hip_roll_joint', 'right_hip_roll_joint',
    'waist_yaw_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
    'waist_pitch_joint', 'left_hip_pitch_joint', 'right_hip_pitch_joint',
    'waist_roll_joint',
    'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
    'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_pitch_joint', 'right_elbow_pitch_joint'
]
# ---------------------------------------------------------------------------- #

URDF = "Wiki-GRx-Models-master/GRX/GR1/GR1T2/urdf/GR1T2_nohand.urdf"
RESOLUTION = (1280, 960)
FPS = 60

def build_scene(urdf_path: Path, show_viewer: bool = True):
    scene = gs.Scene(
        show_viewer=show_viewer,
        viewer_options=gs.options.ViewerOptions(
            res=RESOLUTION,
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=120,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=1.0,
            show_link_frame=False,
            show_cameras=False,
            plane_reflection=True,
            ambient_light=(0.1, 0.1, 0.1),
        ),
        renderer=gs.renderers.Rasterizer()
    )

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(urdf_path),
            fixed=False,
            visualization=True,
            collision=True,
            requires_jac_and_IK=False,
            scale=1.0,
        )
    )

    cam = scene.add_camera(
        res=RESOLUTION,
        pos=(5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
        GUI=show_viewer  # show the camera window when viewer is on
    )

    _ = scene.add_entity(gs.morphs.Plane(), vis_mode="collision")

    scene.build()
    return scene, robot, cam

def run_one_csv(csv_path: Path, video_out: Path,
                scene, robot, cam,
                dof_names=DOF_NAMES,
                base_height: float = 0.0,
                fps: int = FPS):
    df = pd.read_csv(csv_path)
    print("starting", csv_path.name, "->", video_out.name)
    dof_idx_map = {name: i for i, name in enumerate(dof_names)}

    cam.start_recording()
    robot.set_pos([0, 0, 0])

    qpos = np.zeros(robot.get_qpos().shape[0])

    for _, row in df.iterrows():
        # qpos[0:3] = row[['x', 'y', 'z']].values
        # qpos[2] += base_height
        # qpos[3:7] = row[['qw', 'qx', 'qy', 'qz']].values

        qpos[0:3]   = [0.0, 0.0, 0.9]
        qpos[3:7]   = [1, 0.0, 0, 0]
        # qpos[0:3] = row[['x', 'y', 'z']].values
        # qpos[3:7] = row[['qw', 'qx', 'qy', 'qz']].values

        for name, idx in dof_idx_map.items():
            if name in row:
                qpos[idx + 7] = row[name]
        # print(qpos)
        robot.set_qpos(qpos)
        scene.step()
        cam.render()

    cam.stop_recording(save_to_filename=str(video_out), fps=fps)

def main():
    parser = argparse.ArgumentParser(
        description="Run Genesis sim on all angle CSVs in a directory and save MP4s."
    )
    parser.add_argument("--angle_dir", help="Directory containing angle CSVs")
    parser.add_argument("--out_dir", help="Directory to save MP4 videos")
    args = parser.parse_args()

    in_dir = Path(args.angle_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gs.init(backend=gs.gpu)

    scene, robot, cam = build_scene(Path(URDF), show_viewer=True)

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {in_dir}")
        return

    for csv_file in csv_files:
        print(f"[INFO] Processing {csv_file.name}")
        video_out = out_dir / f"{csv_file.stem}.mp4"
        run_one_csv(csv_file, video_out, scene, robot, cam)
        print(f"[DONE] Saved {video_out}")

    print("All files processed.")

if __name__ == "__main__":
    main()
