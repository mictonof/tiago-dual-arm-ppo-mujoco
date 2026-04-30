import csv
from pathlib import Path
import numpy as np
import mujoco

from tiago_right_arm_reach_env import TiagoRightArmReachEnv


OUTDIR = Path("~/projects/tiago_mujoco/diagnostics").expanduser()
OUTDIR.mkdir(parents=True, exist_ok=True)


def vec_str(v):
    return np.array2string(np.asarray(v), precision=4, suppress_small=True)


def get_body_pos(data, body_id):
    return data.xpos[body_id].copy()


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def actuator_report(env):
    print_section("ACTUADORES DEL BRAZO DERECHO")
    rows = []

    for aid in env.right_actuator_ids:
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        lo, hi = env.model.actuator_ctrlrange[aid]
        rows.append([aid, name, lo, hi])
        print(f"[{aid}] {name:35s} ctrlrange=({lo:.4f}, {hi:.4f})")

    write_csv(
        OUTDIR / "right_arm_actuators.csv",
        ["actuator_id", "name", "ctrl_low", "ctrl_high"],
        rows,
    )


def pose_report(env):
    print_section("POSE ACTUAL DEL ROBOT")
    ee = env._ee_pos()
    torso = get_body_pos(env.data, env.torso_body_id)
    shoulder = get_body_pos(env.data, env.right_shoulder_body_id)

    print("Torso      :", vec_str(torso))
    print("Shoulder   :", vec_str(shoulder))
    print("Gripper/EE :", vec_str(ee))
    print("Goal       :", vec_str(env.goal))
    print("Dist EE->Goal      :", np.linalg.norm(env.goal - ee))
    print("Dist Shoulder->Goal:", np.linalg.norm(env.goal - shoulder))
    print("Dist Torso->Goal   :", np.linalg.norm(env.goal - torso))


def sample_goal_distribution(env, n=100):
    print_section(f"DISTRIBUCION DEL GOAL EN {n} RESETS")
    rows = []

    ee_dists = []
    torso_dists = []
    shoulder_dists = []
    goals = []

    for i in range(n):
        obs, info = env.reset(seed=i + 1)
        ee = env._ee_pos()
        torso = get_body_pos(env.data, env.torso_body_id)
        shoulder = get_body_pos(env.data, env.right_shoulder_body_id)
        g = env.goal.copy()

        d_ee = float(np.linalg.norm(g - ee))
        d_torso = float(np.linalg.norm(g - torso))
        d_shoulder = float(np.linalg.norm(g - shoulder))

        goals.append(g)
        ee_dists.append(d_ee)
        torso_dists.append(d_torso)
        shoulder_dists.append(d_shoulder)

        rows.append([
            i,
            g[0], g[1], g[2],
            ee[0], ee[1], ee[2],
            torso[0], torso[1], torso[2],
            shoulder[0], shoulder[1], shoulder[2],
            d_ee, d_torso, d_shoulder
        ])

    goals = np.asarray(goals)
    ee_dists = np.asarray(ee_dists)
    torso_dists = np.asarray(torso_dists)
    shoulder_dists = np.asarray(shoulder_dists)

    print("Goal x min/max/mean:", goals[:, 0].min(), goals[:, 0].max(), goals[:, 0].mean())
    print("Goal y min/max/mean:", goals[:, 1].min(), goals[:, 1].max(), goals[:, 1].mean())
    print("Goal z min/max/mean:", goals[:, 2].min(), goals[:, 2].max(), goals[:, 2].mean())
    print("Dist EE->Goal      min/max/mean:", ee_dists.min(), ee_dists.max(), ee_dists.mean())
    print("Dist Torso->Goal   min/max/mean:", torso_dists.min(), torso_dists.max(), torso_dists.mean())
    print("Dist Shoulder->Goal min/max/mean:", shoulder_dists.min(), shoulder_dists.max(), shoulder_dists.mean())

    write_csv(
        OUTDIR / "goal_distribution.csv",
        [
            "episode",
            "goal_x", "goal_y", "goal_z",
            "ee_x", "ee_y", "ee_z",
            "torso_x", "torso_y", "torso_z",
            "shoulder_x", "shoulder_y", "shoulder_z",
            "dist_ee_goal", "dist_torso_goal", "dist_shoulder_goal",
        ],
        rows,
    )


def actuator_sensitivity(env, delta=0.03, settle_steps=40):
    print_section("SENSIBILIDAD DE CADA ACTUADOR (como mueve el gripper)")
    rows = []

    obs, info = env.reset(seed=123)
    base_ee = env._ee_pos().copy()
    print("EE base:", vec_str(base_ee))

    original_ctrl = env.ctrl.copy()

    for local_idx, aid in enumerate(env.right_actuator_ids):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        lo, hi = env.model.actuator_ctrlrange[aid]
        home = env.ctrl[aid]

        # prueba positiva
        env.data.qpos[:] = env.init_qpos
        env.data.qvel[:] = env.init_qvel
        env.ctrl = original_ctrl.copy()
        env.data.ctrl[:] = env.ctrl
        env._hold_reference()
        mujoco.mj_forward(env.model, env.data)

        plus_target = np.clip(home + delta, lo, hi)
        env.ctrl[aid] = plus_target

        for _ in range(settle_steps):
            env.data.ctrl[:] = env.ctrl
            mujoco.mj_step(env.model, env.data)
            env._hold_reference()
            mujoco.mj_forward(env.model, env.data)

        ee_plus = env._ee_pos().copy()
        d_plus = ee_plus - base_ee
        norm_plus = float(np.linalg.norm(d_plus))

        # prueba negativa
        env.data.qpos[:] = env.init_qpos
        env.data.qvel[:] = env.init_qvel
        env.ctrl = original_ctrl.copy()
        env.data.ctrl[:] = env.ctrl
        env._hold_reference()
        mujoco.mj_forward(env.model, env.data)

        minus_target = np.clip(home - delta, lo, hi)
        env.ctrl[aid] = minus_target

        for _ in range(settle_steps):
            env.data.ctrl[:] = env.ctrl
            mujoco.mj_step(env.model, env.data)
            env._hold_reference()
            mujoco.mj_forward(env.model, env.data)

        ee_minus = env._ee_pos().copy()
        d_minus = ee_minus - base_ee
        norm_minus = float(np.linalg.norm(d_minus))

        print(f"{name:35s} | +delta move={norm_plus:.4f}  d={vec_str(d_plus)} | -delta move={norm_minus:.4f} d={vec_str(d_minus)}")

        rows.append([
            aid, name, home, lo, hi,
            plus_target, d_plus[0], d_plus[1], d_plus[2], norm_plus,
            minus_target, d_minus[0], d_minus[1], d_minus[2], norm_minus,
        ])

    write_csv(
        OUTDIR / "actuator_sensitivity.csv",
        [
            "actuator_id", "name", "home", "ctrl_low", "ctrl_high",
            "plus_target", "plus_dx", "plus_dy", "plus_dz", "plus_norm",
            "minus_target", "minus_dx", "minus_dy", "minus_dz", "minus_norm",
        ],
        rows,
    )


def approximate_workspace(env, n=300):
    print_section(f"APROXIMACION DEL WORKSPACE DEL GRIPPER ({n} muestras)")
    rows = []

    ee_points = []

    # partimos desde reset estable
    env.reset(seed=999)

    base_ctrl = env.ctrl.copy()

    for i in range(n):
        env.data.qpos[:] = env.init_qpos
        env.data.qvel[:] = env.init_qvel
        env.ctrl = base_ctrl.copy()

        # muestreamos targets aleatorios dentro del ctrlrange del brazo derecho
        for aid in env.right_actuator_ids:
            lo, hi = env.model.actuator_ctrlrange[aid]
            env.ctrl[aid] = np.random.uniform(lo, hi)

        env.data.ctrl[:] = env.ctrl
        env._hold_reference()
        mujoco.mj_forward(env.model, env.data)

        for _ in range(60):
            env.data.ctrl[:] = env.ctrl
            mujoco.mj_step(env.model, env.data)
            env._hold_reference()
            mujoco.mj_forward(env.model, env.data)

        ee = env._ee_pos().copy()
        ee_points.append(ee)
        rows.append([i, ee[0], ee[1], ee[2]])

    ee_points = np.asarray(ee_points)

    print("EE x min/max/mean:", ee_points[:, 0].min(), ee_points[:, 0].max(), ee_points[:, 0].mean())
    print("EE y min/max/mean:", ee_points[:, 1].min(), ee_points[:, 1].max(), ee_points[:, 1].mean())
    print("EE z min/max/mean:", ee_points[:, 2].min(), ee_points[:, 2].max(), ee_points[:, 2].mean())

    write_csv(
        OUTDIR / "approx_workspace.csv",
        ["sample", "ee_x", "ee_y", "ee_z"],
        rows,
    )


def main():
    env = TiagoRightArmReachEnv(render_mode=None)

    print_section("ARCHIVOS DE SALIDA")
    print("Se guardarán en:", OUTDIR)

    obs, info = env.reset(seed=42)

    actuator_report(env)
    pose_report(env)
    sample_goal_distribution(env, n=120)
    actuator_sensitivity(env, delta=0.03, settle_steps=40)
    approximate_workspace(env, n=300)

    env.close()

    print_section("LISTO")
    print("Revisa estos archivos:")
    print("-", OUTDIR / "right_arm_actuators.csv")
    print("-", OUTDIR / "goal_distribution.csv")
    print("-", OUTDIR / "actuator_sensitivity.csv")
    print("-", OUTDIR / "approx_workspace.csv")


if __name__ == "__main__":
    main()
