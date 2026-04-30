import time
import numpy as np
import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description

model = load_robot_description("tiago++_mj_description")
data = mujoco.MjData(model)

# Para que no "caiga"
model.opt.gravity[:] = [0.0, 0.0, 0.0]

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        base_pos = data.qpos[:3].copy() if model.nq >= 3 else np.array([0.0, 0.0, 0.8])

        with viewer.lock():
            # Cámara
            viewer.cam.lookat[:] = [base_pos[0], base_pos[1], 0.9]
            viewer.cam.distance = 3.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20

            # 2 geoms extra: fondo blanco + piso claro
            viewer.user_scn.ngeom = 2

            # Fondo blanco grande detrás del robot
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([0.05, 8.0, 5.0]),
                pos=np.array([-2.8, 0.0, 2.0]),
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 1.0, 1.0, 1.0]),
            )

            # Piso claro
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([6.0, 6.0, 0.02]),
                pos=np.array([0.0, 0.0, -0.02]),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.96, 0.96, 0.96, 1.0]),
            )

        viewer.sync()
        time.sleep(0.01)
