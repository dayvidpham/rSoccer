import time
import multiprocessing as mp

import gymnasium as gym
import rsoccer_gym  # noqa: F401 (needed to register envs)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

ENV_ID = "SSLStaticDefenders-v0"
N_ENVS = 64                      # start with 4; if it slows, drop to 2
TOTAL_SECONDS = 5 * 60 * 60     # 5 hours
SAVE_EVERY = 250_000            # timesteps per checkpoint


def make_env(env_id: str):
    def _thunk():
        env = gym.make(env_id)  # no render during training
        return Monitor(env)
    return _thunk


def main():
    # Build vectorized env
    venv = SubprocVecEnv([make_env(ENV_ID) for _ in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        n_steps=256,
        batch_size=32,
        device='cpu'
    )

    start = time.time()
    timesteps = 0

    try:
        while time.time() - start < TOTAL_SECONDS:
            model.learn(total_timesteps=SAVE_EVERY, reset_num_timesteps=False)
            timesteps += SAVE_EVERY
            model.save(f"ppo_{ENV_ID}_{timesteps}")
            venv.save(f"vecnorm_{ENV_ID}.pkl")
    finally:
        # Ensure subprocesses are cleaned up properly
        venv.close()

    print("Done.")


if __name__ == "__main__":
    mp.freeze_support()

    # On macOS, multiprocessing uses "spawn". Being explicit avoids weirdness.
    mp.set_start_method("spawn", force=True)

    main()
