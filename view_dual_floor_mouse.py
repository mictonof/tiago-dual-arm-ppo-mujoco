import time
import numpy as np
import mujoco
import mujoco.viewer
from tiago_dual_arm_reach_env import TiagoDualArmReachEnv

env = TiagoDualArmReachEnv(render_mode=None)
obs, info = env.reset()

print("Brazo activo:", info["arm"])
print("Goal:", info["goal"])

viewer = mujoco.viewer.launch_passive(
    env.model,
    env.data,
    show_left_ui=False,
    show_right_ui=False,
)

camera_initialized = False

try:
    while viewer.is_running():
        env.data.ctrl[:] = env.ctrl
        mujoco.mj_step(env.model, env.data)
        env._hold_reference()
        mujoco.mj_forward(env.model, env.data)

        with viewer.lock():
            viewer.user_scn.ngeom = 3

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
                size=np.array([0.025, 0.0, 0.0]),
                pos=env.goal,
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.1, 0.1, 1.0]),
            )

            if not camera_initialized:
                viewer.cam.lookat[:] = [0.0, 0.0, 0.60]
                viewer.cam.distance = 1.8
                viewer.cam.azimuth = 160
                viewer.cam.elevation = -18
                camera_initialized = True

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
