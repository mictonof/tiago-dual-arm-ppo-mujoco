import time
import numpy as np
import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description

model = load_robot_description("tiago++_mj_description")
data = mujoco.MjData(model)

# Evita que el robot "caiga" o se pierda de la vista
model.opt.gravity[:] = [0.0, 0.0, 0.0]

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Seguir la base del robot con la cámara
        base_pos = data.qpos[:3].copy() if model.nq >= 3 else np.array([0.0, 0.0, 0.8])

        with viewer.lock():
            viewer.cam.lookat[:] = [base_pos[0], base_pos[1], 0.9]
            viewer.cam.distance = 3.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
