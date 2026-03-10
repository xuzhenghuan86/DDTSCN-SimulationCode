"""Train a PPO agent for the FDT global policy using Stable-Baselines3."""

from stable_baselines3 import PPO
from satellite_env import SatelliteEnv

def main():
    print("Initializing FDT PPO Agent...")
    env = SatelliteEnv(task_arrival_rate=200)
    policy_kwargs = dict(net_arch=[128, 128, 64])
    model = PPO("MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=8192,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                policy_kwargs=policy_kwargs)

    # Large-scale training (300k steps for proper convergence)
    print("Starting PPO Training for FDT (300k steps)...")
    model.learn(total_timesteps=300000)

    model.save("ppo_fdt")
    print("Optimization finished. Model saved to ppo_fdt.zip")

if __name__ == '__main__':
    main()
