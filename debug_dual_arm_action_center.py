import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from tiago_arm_reach_env import TiagoArmReachEnv

arm = "right"
if len(sys.argv) > 1:
    arm = sys.argv[1]

env = TiagoArmReachEnv(arm=arm, render_mode=None)
obs, info = env.reset()

def base_pos():
    return env._gripper_base_pos().copy()

def action_center_pos():
    return env._gripper_action_center_pos().copy()

viewer = mujoco.viewer.launch_passive(
    env.model, env.data,
    show_left_ui=False,
    show_right_ui=False,
)

try:
    while viewer.is_running():
        env.data.ctrl[:] = env.ctrl
        mujoco.mj_step(env.model, env.data)
        env._hold_reference()
        mujoco.mj_forward(env.model, env.data)

        b = base_pos()
        c = action_center_pos()
        g = env.goal.copy()

        with viewer.lock():
            viewer.user_scn.ngeom = 6

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([0.03, 4.0, 3.0]),
                pos=np.array([-1.8, 0.0, 1.5]),
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 1.0, 1.0, 1.0]),
            )

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([3.0, 3.0, 0.02]),
                pos=np.array([0.0, 0.0, -0.02]),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.96, 0.96, 0.96, 1.0]),
            )

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[2],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.02, 0.0, 0.0]),
                pos=g,
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.1, 0.1, 1.0]),
            )

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[3],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.02, 0.0, 0.0]),
                pos=b,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.1, 0.3, 1.0, 1.0]),
            )

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[4],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.02, 0.0, 0.0]),
                pos=c,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.1, 1.0, 0.2, 1.0]),
            )

            seg = c - b
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-9:
                seg = np.array([1.0, 0.0, 0.0])
                seg_len = 1e-6
            seg_dir = seg / seg_len
            midpoint = 0.5 * (b + c)

            z = np.array([0.0, 0.0, 1.0])
            x = seg_dir
            y = np.cross(z, x)
            if np.linalg.norm(y) < 1e-9:
                y = np.array([0.0, 1.0, 0.0])
            y = y / np.linalg.norm(y)
            z = np.cross(x, y)
            z = z / np.linalg.norm(z)
            mat = np.column_stack([x, y, z]).reshape(-1)

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[5],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.array([0.006, seg_len / 2.0, 0.0]),
                pos=midpoint,
                mat=mat,
                rgba=np.array([1.0, 0.9, 0.1, 1.0]),
            )

            viewer.cam.lookat[:] = [0.0, -0.10 if arm == "right" else 0.10, 0.58]
            viewer.cam.distance = 1.2
            viewer.cam.azimuth = 155
            viewer.cam.elevation = -15

        viewer.sync()
        time.sleep(0.02)

except KeyboardInterrupt:
    pass
finally:
    try:
        viewer.close()
    except Exception:
        pass
    env.close()
