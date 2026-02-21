#!/usr/bin/env python3
import argparse
import sys

import gymnasium as gym
import numpy as np
import rsoccer_gym


def run_episode(env, policy_fn, max_steps=10_000, render=False):
    obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    steps = 0

    while not (terminated or truncated):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        if render:
            # render_mode="human" is handled internally by env,
            # but some gym envs still require explicit render calls.
            try:
                env.render()
            except Exception:
                pass

        if steps >= max_steps:
            # safety stop in case env never terminates
            break

    return total_reward, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SSLDribbling-v0", help="Gym env id (e.g., VSS-v0, SSLDribbling-v0)")
    parser.add_argument("--render", action="store_true", help="Enable rendering (human)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10_000)

    # Policy selection
    parser.add_argument("--policy", choices=["random", "ppo"], default="random")
    parser.add_argument("--model-path", type=str, default=None, help="Path to SB3 model zip (required for --policy ppo)")

    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = gym.make(args.env, render_mode=render_mode)

    print(f"Env: {args.env}")
    print(f"Obs space: {env.observation_space}")
    print(f"Act space: {env.action_space}")

    # --- build policy function ---
    if args.policy == "random":
        def policy_fn(_obs):
            return env.action_space.sample()

    elif args.policy == "ppo":
        if not args.model_path:
            print("ERROR: --model-path is required when --policy ppo", file=sys.stderr)
            sys.exit(1)

        from stable_baselines3 import PPO

        model = PPO.load(args.model_path, device="cpu")

        # sanity check: action dim
        # SB3 policies output action with same shape as env.action_space.shape
        exp_shape = getattr(env.action_space, "shape", None)
        if exp_shape is None:
            print("WARNING: action space has no .shape; skipping shape check")
        else:
            # quick probe
            obs0, _ = env.reset()
            act0, _ = model.predict(obs0, deterministic=True)
            act0 = np.array(act0)
            if act0.shape != exp_shape:
                print(
                    "WARNING: model action shape does not match env action shape!\n"
                    f"  model action shape: {act0.shape}\n"
                    f"  env action shape:   {exp_shape}\n"
                    "This usually means you trained on a different env (e.g., VSS-v0 vs SSLDribbling-v0).\n"
                    "Expect errors or nonsense behavior."
                )

        def policy_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

    else:
        raise RuntimeError("Unknown policy")

    # --- run episodes ---
    for ep in range(args.episodes):
        total_reward, steps = run_episode(env, policy_fn, max_steps=args.max_steps, render=args.render)
        print(f"Episode {ep+1}: steps={steps}, total_reward={total_reward}")

    env.close()


if __name__ == "__main__":
    main()
