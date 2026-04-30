from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from tiago_dual_arm_reach_env import TiagoDualArmReachEnv

train_env = Monitor(TiagoDualArmReachEnv(render_mode=None))
eval_env = Monitor(TiagoDualArmReachEnv(render_mode=None))

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path="./checkpoints_dual",
    name_prefix="ppo_dual_auto_arm_precise"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_dual_model",
    log_path="./eval_logs_dual",
    eval_freq=10_000,
    deterministic=True,
    render=False,
    n_eval_episodes=20,
)

policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    device="cpu",
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.02,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./tb_dual_ppo",
    verbose=1,
)

model.learn(
    total_timesteps=500_000,
    callback=[checkpoint_callback, eval_callback],
)

model.save("ppo_tiago_dual_auto_arm_reach_precise_500k")
train_env.close()
eval_env.close()
