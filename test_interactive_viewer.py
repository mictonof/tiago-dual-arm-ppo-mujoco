import mujoco
import mujoco.viewer
from robot_descriptions.loaders.mujoco import load_robot_description

model = load_robot_description("tiago++_mj_description")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# visor bloqueante e interactivo
mujoco.viewer.launch(model, data)
