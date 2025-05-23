import cv2
from envs.sumo_env1 import SumoEnv

env = SumoEnv(render=True)
env._external_preview = True  #  Tell the env not to do internal preview
obs, _ = env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)

    env._preview_camera(env.botA, env.head_idx_A, "Bot A Cam")
    env._preview_camera(env.botB, env.head_idx_B, "Bot B Cam")
    cv2.waitKey(1)

    if done or truncated:
        obs, _ = env.reset()

