import gymnasium as gym
import rsoccer_gym

# Minimal test script for Docker demo (video-style)
def main():
    env = gym.make('VSS-v0', render_mode=None)
    env.reset()
    total_reward = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample() # Random policy
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    print(f"Episode finished. Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()
