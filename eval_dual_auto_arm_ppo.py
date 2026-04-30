import numpy as np
from stable_baselines3 import PPO
from tiago_dual_arm_reach_env import TiagoDualArmReachEnv

env = TiagoDualArmReachEnv(render_mode="human")
model = PPO.load("./best_dual_model/best_model", env=env, device="cpu")

VISUAL_CLOSE_THRESHOLD = 0.05

while True:
    obs, info = env.reset()
    print("Brazo activo:", info["arm"], "Goal:", info["goal"])

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            ee = env._ee_pos(env.active_arm).copy()
            goal = env.goal.copy()
            visual_dist = float(np.linalg.norm(ee - goal))
            visual_success = visual_dist <= VISUAL_CLOSE_THRESHOLD

            print({
                **info,
                "visual_distance": visual_dist,
                "visual_success": visual_success,
            })
            break
