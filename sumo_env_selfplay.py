import os
import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_data
from gymnasium import spaces
from math import pi
import random
import torch

class SumoEnvSelfPlay(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render=False, opponent_policy=None):
        super().__init__()
        self.render = render
        self.time_step = 1/60
        self.preview_rate = 2
        self.ring_radius = 1.0
        self.max_steps = 1500
        self.fall_threshold = 1.5
        self.episode_return = 0.0

        self.opponent_policy = opponent_policy  # function or callable

        if self.render:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)  # Only botA
        high = np.inf * np.ones(14, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        self.episode_return = 0.0

        p.loadURDF("plane.urdf")
        black = p.createVisualShape(p.GEOM_CYLINDER, radius=self.ring_radius, length=0.01,
                                    rgbaColor=[0, 0, 0, 1], visualFramePosition=[0, 0, 0.005])
        p.createMultiBody(0, baseVisualShapeIndex=black)
        white = p.createVisualShape(p.GEOM_CYLINDER, radius=self.ring_radius * 1.03, length=0.01,
                                    rgbaColor=[1, 1, 1, 1], visualFramePosition=[0, 0, 0.01])
        p.createMultiBody(0, baseVisualShapeIndex=white)

        urdf = os.path.join(os.path.dirname(__file__), "robot_model.urdf")
        yaw_opts = [pi/2, -pi/2]
        ornA = p.getQuaternionFromEuler([0, 0, random.choice(yaw_opts)])
        ornB = p.getQuaternionFromEuler([0, 0, random.choice(yaw_opts)])
        self.botA = p.loadURDF(urdf, [-0.5, 0, 0.07], ornA)
        self.botB = p.loadURDF(urdf, [0.5, 0, 0.07], ornB)

        for bot in (self.botA, self.botB):
            for j in range(p.getNumJoints(bot)):
                p.setJointMotorControl2(bot, j, p.VELOCITY_CONTROL, force=0)

        self.head_idx_A = self._find_link(self.botA, 'head')
        self.head_idx_B = self._find_link(self.botB, 'head')
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, actionA):
        actionB = self._get_opponent_action()
        speedsA = np.clip(actionA, -1.0, 1.0) * 10.0
        speedsB = np.clip(actionB, -1.0, 1.0) * 10.0

        for bot, (l, r) in zip((self.botA, self.botB), [(speedsA[0], speedsA[1]), (speedsB[0], speedsB[1])]):
            p.setJointMotorControl2(bot, 0, p.VELOCITY_CONTROL, targetVelocity=l, force=10)
            p.setJointMotorControl2(bot, 1, p.VELOCITY_CONTROL, targetVelocity=r, force=10)

        p.stepSimulation()
        self.step_count += 1

        obs = self._get_obs()
        reward, done = self._compute_reward()
        truncated = (self.step_count >= self.max_steps)
        self.episode_return += reward

        return obs, reward, done, truncated, {}

    def _get_opponent_action(self):
        return self.opponent_policy(self._get_obs())

        # if callable(self.opponent_policy):
        #     obs = self._get_obs()
        #     return self.opponent_policy(obs)
        # return np.random.uniform(-1, 1, size=(2,))

    def _compute_reward(self):
        posA, _ = p.getBasePositionAndOrientation(self.botA)
        posB, _ = p.getBasePositionAndOrientation(self.botB)
        outA = np.linalg.norm(posA[:2]) > self.ring_radius
        outB = np.linalg.norm(posB[:2]) > self.ring_radius

        if outA and not outB:
            rewardA = -1.0
        elif outB and not outA:
            rewardA = +1.0
        elif outA and outB:
            rewardA = -0.2
        else:
            rewardA = 0

        _,ornA = p.getBasePositionAndOrientation(self.botA)
        _,ornB = p.getBasePositionAndOrientation(self.botB)
        rA,pA,_ = p.getEulerFromQuaternion(ornA)
        rB,pB,_ = p.getEulerFromQuaternion(ornB)
        if abs(rA) > self.fall_threshold or abs(pA) > self.fall_threshold:
            rewardA += -1.0
        elif abs(rB) > self.fall_threshold or abs(pB) > self.fall_threshold:
            rewardA += 1

        # shaping reward by distant
        dist = np.linalg.norm(np.array(posB[:2])-np.array(posA[:2]))
        dist_reward = (1-dist/self.ring_radius)*0.01 *100  # closer opponent
        + (np.linalg.norm(posB[:2])/self.ring_radius)*0.01 *1000 #
        - 0.001
        
        total_step_reward = 1 * rewardA
        - 1* dist_reward


        return total_step_reward, (rewardA != 0)

    def _get_obs(self):
        posA, ornA = p.getBasePositionAndOrientation(self.botA)
        velA, _ = p.getBaseVelocity(self.botA)
        posB, ornB = p.getBasePositionAndOrientation(self.botB)
        velB, _ = p.getBaseVelocity(self.botB)
        vec = np.array(posB[:2]) - np.array(posA[:2])
        dist = np.linalg.norm(vec) + 1e-8
        bearing = np.arctan2(vec[1], vec[0])
        rA, pA, _ = p.getEulerFromQuaternion(ornA)
        rB, pB, _ = p.getEulerFromQuaternion(ornB)
        return np.array([
            posA[0], posA[1], velA[0], velA[1],
            posB[0], posB[1], velB[0], velB[1],
            dist, bearing, rA, pA, rB, pB
        ], dtype=np.float32)

    def _find_link(self, body, name):
        for i in range(p.getNumJoints(body)):
            if p.getJointInfo(body, i)[12].decode() == name:
                return i
        return -1
    
    def _preview_camera(self, bot, link_idx, label="Cam"):
        import cv2
        state = p.getLinkState(bot, link_idx, computeForwardKinematics=True)
        pos, orn = state[0], state[1]
        rot = p.getMatrixFromQuaternion(orn)
        forward = np.array([rot[0], rot[3], rot[6]])
        up      = np.array([rot[2], rot[5], rot[8]])
        eye    = pos + up*1
        target = pos + forward*0.10
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.01, 2.0)
        _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_np = np.reshape(rgb, (512, 512, 4))[:, :, :3].astype(np.uint8)
        cv2.imshow(label, rgb_np)
        return rgb_np

    def render(self): pass

    def close(self): p.disconnect(self.physics_client)
