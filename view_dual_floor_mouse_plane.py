import time
import numpy as np
import mujoco
import mujoco.viewer
from tiago_dual_arm_reach_env import TiagoDualArmReachEnv

env = TiagoDualArmReachEnv(render_mode=None)
obs, info = env.reset()

print("Brazo activo:", info["arm"])
print("Goal:", info["goal"])
print("Teclas:")
print("  P -> mostrar/ocultar plano central")
print("  Z -> mostrar/ocultar prisma")
print("  G -> mostrar/ocultar puntos verdes")
print("  R -> reset")

state = {
    "show_plane": True,
    "show_zone": True,
    "show_grippers": True,
    "camera_initialized": False,
    "do_reset": False,
}

def key_callback(keycode):
    try:
        key = chr(keycode).lower()
    except ValueError:
        return
    if key == "p":
        state["show_plane"] = not state["show_plane"]
    elif key == "z":
        state["show_zone"] = not state["show_zone"]
    elif key == "g":
        state["show_grippers"] = not state["show_grippers"]
    elif key == "r":
        state["do_reset"] = True

viewer = mujoco.viewer.launch_passive(
    env.model,
    env.data,
    key_callback=key_callback,
    show_left_ui=False,
    show_right_ui=False,
)

try:
    while viewer.is_running():
        if state["do_reset"]:
            obs, info = env.reset()
            print("Brazo activo:", info["arm"], "Goal:", info["goal"])
            state["do_reset"] = False

        env.data.ctrl[:] = env.ctrl
        mujoco.mj_step(env.model, env.data)
        env._hold_reference()
        mujoco.mj_forward(env.model, env.data)

        torso = env.data.xpos[env.torso_body_id].copy()
        goal = env.goal.copy()
        ee_right = env._ee_pos("right").copy()
        ee_left = env._ee_pos("left").copy()
        zone_center = env.get_sampling_prism_center().copy()
        zone_half = env.get_sampling_prism_half_extents().copy()

        ngeom = 3
        if state["show_zone"]:
            ngeom += 1
        if state["show_plane"]:
            ngeom += 1
        if state["show_grippers"]:
            ngeom += 2

        with viewer.lock():
            viewer.user_scn.ngeom = ngeom
            idx = 0

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[idx],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([0.03, 4.0, 3.0]),
                pos=np.array([-1.8, 0.0, 1.5]),
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 1.0, 1.0, 1.0]),
            )
            idx += 1

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[idx],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([3.0, 3.0, 0.02]),
                pos=np.array([0.0, 0.0, -0.02]),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.96, 0.96, 0.96, 1.0]),
            )
            idx += 1

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([env.goal_radius, 0.0, 0.0]),
                pos=goal,
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.1, 0.1, 1.0]),
            )
            idx += 1

            if state["show_zone"]:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[idx],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=zone_half,
                    pos=zone_center,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0.80, 0.80, 0.80, 0.07]),
                )
                idx += 1

            if state["show_plane"]:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[idx],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=np.array([zone_half[0], 0.0015, zone_half[2]]),
                    pos=np.array([zone_center[0], torso[1], zone_center[2]], dtype=np.float64),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0.55, 0.20, 0.85, 0.10]),
                )
                idx += 1

            if state["show_grippers"]:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[idx],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([env.ee_visual_radius, 0.0, 0.0]),
                    pos=ee_right,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0.0, 0.85, 0.20, 1.0]),
                )
                idx += 1

                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[idx],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([env.ee_visual_radius, 0.0, 0.0]),
                    pos=ee_left,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0.15, 1.0, 0.35, 1.0]),
                )
                idx += 1

            if not state["camera_initialized"]:
                viewer.cam.lookat[:] = [0.30, 0.0, 0.58]
                viewer.cam.distance = 1.55
                viewer.cam.azimuth = 160
                viewer.cam.elevation = -16
                state["camera_initialized"] = True

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
