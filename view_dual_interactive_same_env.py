import mujoco
import mujoco.viewer
from tiago_dual_arm_reach_env import TiagoDualArmReachEnv

env = TiagoDualArmReachEnv(render_mode=None)

obs, info = env.reset(seed=42)
print("Brazo activo:", info["arm"])
print("Goal:", info["goal"])

# Ya usas gravedad cero dentro del entorno
# y la pose home ya quedó aplicada en reset()

mujoco.mj_forward(env.model, env.data)

# Visor interactivo estándar
mujoco.viewer.launch(env.model, env.data)

env.close()
