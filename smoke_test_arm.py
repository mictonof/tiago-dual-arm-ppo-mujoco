import time
import numpy as np
from tiago_right_arm_reach_env import TiagoRightArmReachEnv

env = TiagoRightArmReachEnv(render_mode="human")

try:
    obs, info = env.reset(seed=42)

    for _ in range(300):
        action = np.random.uniform(-0.3, 0.3, size=(7,))
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.02)

        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()
