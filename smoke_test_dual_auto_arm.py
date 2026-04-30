import time
import numpy as np
from tiago_dual_arm_reach_env import TiagoDualArmReachEnv

env = TiagoDualArmReachEnv(render_mode="human")

try:
    obs, info = env.reset(seed=42)
    print("Active arm:", info["arm"], "Goal:", info["goal"])

    for _ in range(400):
        action = np.random.uniform(-0.3, 0.3, size=(7,))
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.02)

        if terminated or truncated:
            obs, info = env.reset()
            print("Active arm:", info["arm"], "Goal:", info["goal"])
finally:
    env.close()
