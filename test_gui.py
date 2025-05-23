import cv2
from stable_baselines3 import PPO
from envs.sumo_env1 import SumoEnv
#from envs.sumo_env2 import SumoEnv
#from envs.sumo_env3 import SumoEnv
#from envs.sumo_env4 import SumoEnv

# Create environment with GUI
env = SumoEnv(render=True)
env._external_preview = True  #  Prevent internal cam previews

# Load trained model
model = PPO.load("checkpoints_v1(backup)\ppo_sumo_2000000.zip", env=env)

# Reset env
obs, _ = env.reset()

while True:
    #  Display both robot head cams
    env._preview_camera(env.botA, env.head_idx_A, "Bot A Cam")
    env._preview_camera(env.botB, env.head_idx_B, "Bot B Cam")
    cv2.waitKey(1)

    # Predict and step using PPO model
    act, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(act)

    if done:
        obs, _ = env.reset()

