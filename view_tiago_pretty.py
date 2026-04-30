import time
import numpy as np
import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description

model = load_robot_description("tiago++_mj_description")
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(
    model,
    data,
    show_left_ui=False,
    show_right_ui=False,
) as viewer:
    with viewer.lock():
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.9])
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -18

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
