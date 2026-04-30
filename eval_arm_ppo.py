from stable_baselines3 import PPO
from tiago_right_arm_reach_env import TiagoRightArmReachEnv

env = TiagoRightArmReachEnv(render_mode="human")
model = PPO.load("ppo_tiago_right_arm_reach", env=env, device="cpu")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(info)
        obs, info = env.reset()
