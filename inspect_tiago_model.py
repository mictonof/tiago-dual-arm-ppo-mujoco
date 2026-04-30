import mujoco
from robot_descriptions.loaders.mujoco import load_robot_description

model = load_robot_description("tiago++_mj_description")

print("\n=== JOINTS ===")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"{i:3d} | {name}")

print("\n=== ACTUATORS ===")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"{i:3d} | {name}")

print("\n=== BODIES ===")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"{i:3d} | {name}")

print("\n=== SITES ===")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"{i:3d} | {name}")
