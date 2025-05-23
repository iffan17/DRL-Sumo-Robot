import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from envs.sumo_env3 import SumoEnv

class SaveAndLogCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_counts = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # This callback is called after each call to env.step()
        # Check for done signals in the vectorized env
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                if "episode" in info.keys():
                    self.episode_count += 1
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_counts.append(self.episode_count)
                    if self.verbose > 0:
                        print(f"Episode {self.episode_count} reward: {info['episode']['r']}")
        
        # Save model every save_freq timesteps
        if (self.num_timesteps % self.save_freq) == 0:
            save_file = os.path.join(self.save_path, f"ppo_sumo_{self.num_timesteps}")
            if self.verbose > 0:
                print(f"Saving model checkpoint to {save_file}")
            self.model.save(save_file)

        return True

if __name__=="__main__":
    save_dir = "./checkpoints_v3"
    os.makedirs(save_dir, exist_ok=True)

    env = make_vec_env(lambda: SumoEnv(render=False), n_envs=8)

    model = PPO("MlpPolicy", env,
                tensorboard_log="./logs",
                device="cuda",
                n_steps=2048,
                batch_size=4096,
                learning_rate=3e-4,
                n_epochs=10)

    # Callback to save every 500k timesteps
    save_freq = 500_000
    callback = SaveAndLogCallback(save_freq=save_freq, save_path=save_dir)

    # Train for 2 million timesteps with callback
    model.learn(total_timesteps=10_000_000, callback=callback)

    # Save final model
    model.save(os.path.join(save_dir, "ppo_sumo_final"))

    # Plot episode rewards
    plt.plot(callback.episode_counts, callback.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Episode Rewards")
    plt.grid(True)

    # Save plot image file alongside checkpoints
    plot_path = os.path.join(save_dir, "training_rewards.png")
    plt.savefig(plot_path)   # Save the figure as PNG file
    print(f"Saved training rewards plot to: {plot_path}")

    plt.show()


    # Example: How to load the saved policy
    # from stable_baselines3 import PPO
    # model = PPO.load("./checkpoints/ppo_sumo_2000000")
    # model.set_env(env)
