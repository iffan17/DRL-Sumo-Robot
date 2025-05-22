import os
import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_data
from gymnasium import spaces
from math import sqrt, pi
# at top of your script
import cv2

# version note :  right now the env is resetting everytime an agent pitch/row too high
# it's helping prevent agent glitch and flying out but might reduce possible move for agent
class SumoEnv(gym.Env):
    """
    2-agent Sumo environment with head-mounted RGB camera previews at 30 FPS.
    Observations: 14-d state vector (positions, velocities, roll/pitch, distance, bearing).
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render=False):
        super().__init__()
        self.render = render
        self.time_step = 1/60    # physics at 60Hz
        self.preview_rate = 2     # render camera every 2 steps → 30 FPS
        self.ring_radius = 1.0
        self.max_steps = 1500
        self.fall_threshold = 1.0  # radians
        self.episode_return = 0.0

        # connect
        if self.render:
            self.physics_client = p.connect(p.GUI)
            # only RGB buffer preview
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,       0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,     0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)

        # Action: left/right wheels for both bots
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        # Observation: 14-dim state
        high = np.inf * np.ones(14, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)

        self.episode_return = 0.0

        # ground + ring
        p.loadURDF("plane.urdf")
        black = p.createVisualShape(p.GEOM_CYLINDER, radius=self.ring_radius, length=0.01,
                                    rgbaColor=[0,0,0,1], visualFramePosition=[0,0,0.005])
        p.createMultiBody(0, baseVisualShapeIndex=black)
        white = p.createVisualShape(p.GEOM_CYLINDER, radius=self.ring_radius*1.03, length=0.01,
                                    rgbaColor=[1,1,1,1], visualFramePosition=[0,0,0.01])
        p.createMultiBody(0, baseVisualShapeIndex=white)

        # load robots with random sideways orientation
        urdf = os.path.join(os.path.dirname(__file__), "robot_model.urdf")
        yaw_opts = [pi/2, -pi/2]
        ornA = p.getQuaternionFromEuler([0,0,self.np_random.choice(yaw_opts)])
        ornB = p.getQuaternionFromEuler([0,0,self.np_random.choice(yaw_opts)])
        self.botA = p.loadURDF(urdf, [-0.5,0,0.07], ornA)
        self.botB = p.loadURDF(urdf, [ 0.5,0,0.07], ornB)
        for bot in (self.botA,self.botB):
            for j in range(p.getNumJoints(bot)):
                p.setJointMotorControl2(bot, j, p.VELOCITY_CONTROL, force=0)
        # find head link for camera mount
        self.head_idx_A = self._find_link(self.botA, 'head')
        self.head_idx_B = self._find_link(self.botB, 'head')

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        speeds = np.clip(action, -1.0, 1.0) * 10.0
        for bot, (l, r) in zip((self.botA, self.botB), [(speeds[0], speeds[1]), (speeds[2], speeds[3])]):
            p.setJointMotorControl2(bot, 0, p.VELOCITY_CONTROL, targetVelocity=l, force=10)
            p.setJointMotorControl2(bot, 1, p.VELOCITY_CONTROL, targetVelocity=r, force=10)

        p.stepSimulation()
        self.step_count += 1

        # Only run internal preview if not running from external test
        preview = getattr(self, "_external_preview", False) is False and \
                  self.render and self.step_count % self.preview_rate == 0

        if preview:
            self._preview_camera(self.botA, self.head_idx_A, "Bot A Cam")
            self._preview_camera(self.botB, self.head_idx_B, "Bot B Cam")

        obs = self._get_obs()
        reward, done = self._compute_reward()
        truncated = (self.step_count >= self.max_steps)
        self.episode_return += reward

        return obs, reward, done, truncated, {}

    def _compute_reward(self):
        
        posA,_ = p.getBasePositionAndOrientation(self.botA)
        posB,_ = p.getBasePositionAndOrientation(self.botB)
        outA = np.linalg.norm(posA[:2])>self.ring_radius
        outB = np.linalg.norm(posB[:2])>self.ring_radius

        # Win / Loss
        if outA and not outB:
            knockout_reward = -1.0  # Agent A loses
        elif outB and not outA:
            knockout_reward = +1.0  # Agent A wins
        elif outA and outB:
            knockout_reward = -0.2   # draw / both fall
        else:
            knockout_reward = 0
        
        # fall detection
        _,ornA = p.getBasePositionAndOrientation(self.botA)
        _,ornB = p.getBasePositionAndOrientation(self.botB)
        rA,pA,_ = p.getEulerFromQuaternion(ornA)
        rB,pB,_ = p.getEulerFromQuaternion(ornB)

        # -1 if fall else +1
        if abs(rA)>self.fall_threshold or abs(pA)>self.fall_threshold:
            flip_reward = -1.0
        if abs(rB)>self.fall_threshold or abs(pB)>self.fall_threshold:
            flip_reward = +1.0
        else:
            flip_reward = 0
        
        # shaping reward by distant
        dist = np.linalg.norm(np.array(posB[:2])-np.array(posA[:2]))
        dist_reward = (1-dist/self.ring_radius)*0.01 + (np.linalg.norm(posB[:2])/self.ring_radius)*0.01 - 0.001, False
        
        total_step_reward = 1 * knockout_reward
        + 1* flip_reward
        + 1* dist_reward

        reset = (knockout_reward != 0 ) #or flip_reward != 0
        return total_step_reward, reset

    def _get_obs(self):
        posA,ornA = p.getBasePositionAndOrientation(self.botA)  # obs - p_a
        velA,_ = p.getBaseVelocity(self.botA)                   # obs - v_a
        posB,ornB = p.getBasePositionAndOrientation(self.botB)  # obs - p_b
        velB,_ = p.getBaseVelocity(self.botB)                   # obs - v_B
        vec = np.array(posB[:2]) - np.array(posA[:2])               # (vector between agent)
        dist = np.linalg.norm(vec)+1e-8                         # obs - distant between agent
        bearing = np.arctan2(vec[1], vec[0])                    # obs - ??? angle between agent 
        rA,pA,_ = p.getEulerFromQuaternion(ornA)                # obs - row, pitch A
        rB,pB,_ = p.getEulerFromQuaternion(ornB)                # obs - row, pitch B
        return np.array([
            posA[0],posA[1], velA[0],velA[1],
            posB[0],posB[1], velB[0],velB[1],
            dist, bearing,
            rA, pA, rB, pB
        ], dtype=np.float32)

    def _find_link(self, body, name):
        for i in range(p.getNumJoints(body)):
            if p.getJointInfo(body,i)[12].decode()==name:
                return i
        return -1

    def _preview_camera(self, bot, link_idx, label="Cam"):
        import cv2
        state = p.getLinkState(bot, link_idx, computeForwardKinematics=True)
        pos, orn = state[0], state[1]
        rot = p.getMatrixFromQuaternion(orn)
        forward = np.array([rot[0], rot[3], rot[6]])
        up      = np.array([rot[2], rot[5], rot[8]])
        eye    = pos + up*0.02
        target = pos + forward*0.10
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.01, 2.0)
        _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_np = np.reshape(rgb, (512, 512, 4))[:, :, :3].astype(np.uint8)
        cv2.imshow(label, rgb_np)
        return rgb_np


    def render(self): pass

    def close(self): p.disconnect(self.physics_client)

