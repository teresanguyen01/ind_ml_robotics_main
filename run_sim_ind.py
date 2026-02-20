import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'
import genesis as gs
import numpy as np
import pandas as pd
import torch

# VIDEO_PATH = "1202_toddler_data/videos"
# FILE_NAME = "running_veronica_2_1003_pred_resamp_qpos_smoothed"

FILE_PATH = "/home/tt/ind_ml_rawan_teresa/AA_Yan_mocap_included_0120/model28_toddler_and_adult/aa_pred_filter/experiment3_021626_pred.csv"

# FILE_PATH = "AA-MAIN-FOLDER/YAN_0120_test_aa/Yan_combo3_1.csv"

# FILE_PATH = "AA_SENSOR-RAWAN-VERO-TERE/predictions/aa_pred_filter/Experiment3_Run1_1031_Veronica_pred_resamp_smoothed.csv" 
URDF = "Wiki-GRx-Models-master/GRX/GR1/GR1T2/urdf/GR1T2_nohand.urdf"
gs.init(backend=gs.gpu)

sim_opts = gs.options.SimOptions(gravity=(0.0, -9.81, 0.0))

scene = gs.Scene(
    # sim_options=sim_opts,
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
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
    renderer = gs.renderers.Rasterizer(),
)

# cam_pose = scene.viewer.camera_pose

# scene.viewer.set_camera_pose(cam_pose)

robot = scene.add_entity(
    gs.morphs.URDF(
        file=URDF,
        fixed=False,
        visualization=True,
        collision=True,
        requires_jac_and_IK=False,
        scale=1.0,
    )
)

# cam = scene.add_camera(
#     res=(1280, 960),
#     pos=(5, 0.0, 2.5),
#     lookat=(0, 0, 0.5),
#     fov=30,
#     GUI=False
# )

plane = scene.add_entity(gs.morphs.Plane(), vis_mode="collision")

scene.build()

# rgb, depth, segmentation, normal = cam.render(depth=False, segmentation=False, normal=False)

df = pd.read_csv(FILE_PATH)

dof_names = [
      'left_hip_roll_joint', 'right_hip_roll_joint',
      'waist_yaw_joint', 
      'left_hip_yaw_joint', 'right_hip_yaw_joint',
      'waist_pitch_joint', 
      'left_hip_pitch_joint', 'right_hip_pitch_joint',
      'waist_roll_joint',
      'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
      'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
      'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
      'left_elbow_pitch_joint', 'right_elbow_pitch_joint'
]

dof_idx_map = {name: i for i, name in enumerate(dof_names)}

# cam.start_recording()
robot.set_pos([0, 0, 0])
qpos = np.zeros(robot.get_qpos().shape[0])
y_offset = 0.75
qpos[0:3]   = [0.0, 0.0, 0.9]
# qpos[3:7]   = [1, 0.0, 0, 0]
# qpos[3:7] = [1, 0, 0, 0]
left_leg_link = robot.get_link(name="left_thigh_pitch_link")
right_leg_link = robot.get_link(name="right_thigh_pitch_link")
print(left_leg_link, right_leg_link)
# for joint in robot.joints: 
#     print(f"name: {joint.name}, dof_start: {joint.dof_start}, dof_end: {joint.dof_end}, {joint.idx}, type: {joint.type}")

for _, row in df.iterrows():
    # qpos[0:3] = row[['x', 'y', 'z']].values
    qpos[3:7] = [1, 0.0, 0, 0]
    # qpos[2]  += y_offset
    # if _ == 0:
    #     print(_)
    #     LEFT_SHOULDER_PITCH_OFFSET = row['left_shoulder_pitch_joint']
    #     RIGHT_SHOULDER_PITCH_OFFSET = row['right_shoulder_pitch_joint']
    #     print(LEFT_SHOULDER_PITCH_OFFSET, RIGHT_SHOULDER_PITCH_OFFSET)
    
    for dof_name, dof_idx in dof_idx_map.items():
        if dof_name in row:
            qpos[dof_idx + 7] = row[dof_name] 
        if dof_name == 'left_hip_pitch_joint':
            qpos[dof_idx + 7] = 0
        if dof_name == 'right_hip_pitch_joint':
            qpos[dof_idx + 7] = 0
        if dof_name == 'right_hip_roll_joint':
            qpos[dof_idx + 7] = 0
        if dof_name == 'left_hip_roll_joint':
            qpos[dof_idx + 7] = 0
        if dof_name == 'left_hip_yaw_joint':
            qpos[dof_idx + 7] = 0
        if dof_name == 'right_hip_yaw_joint':
            qpos[dof_idx + 7] = 0
        # if dof_name == 'waist_yaw_joint':
        #     qpos[dof_idx + 7] = row['qz']
        # if dof_name == 'waist_pitch_joint':
        #     qpos[dof_idx + 7] = row['qy']
        # if dof_name == 'waist_roll_joint':
        #     qpos[dof_idx + 7] = row['qx']
    

    robot.set_qpos(qpos)
    # lf_aabb_min, lf_aabb_max = left_leg_link.get_AABB()
    # rf_aabb_min, rf_aabb_max = right_leg_link.get_AABB()

    # # handle possible shapes (3,) or (1, 3)
    # lf_min = lf_aabb_min[0] if lf_aabb_min.ndim == 2 else lf_aabb_min
    # rf_min = rf_aabb_min[0] if rf_aabb_min.ndim == 2 else rf_aabb_min

    # lf_z = lf_min[2].item() if isinstance(lf_min, torch.Tensor) else float(lf_min[2])
    # rf_z = rf_min[2].item() if isinstance(rf_min, torch.Tensor) else float(rf_min[2])

    # min_foot_z = min(lf_z, rf_z)

    # if min_foot_z < 0.0:
    #     qpos[2] -= min_foot_z
    #     robot.set_qpos(qpos)
    
    # print("row number", _)

    # print(qpos)
    scene.step()
    # cam.render()

# cam.stop_recording(save_to_filename=f'{VIDEO_PATH}/{FILE_NAME}.mp4', fps=120)
