from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from tiago_arm_reach_env import TiagoArmReachEnv

env = Monitor(TiagoArmReachEnv(arm="left", render_mode=None))

model = PPO(
    policy="MlpPolicy",
    env=env,
    device="cpu",
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.03,
    verbose=1,
)

model.learn(total_timesteps=300_000)
model.save("ppo_tiago_left_arm_reach")
env.close()
