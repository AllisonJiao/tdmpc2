try:
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import os
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
    import mujoco
    from gripper_controller import rand_spawn
except ImportError as e:
    print("\n GRIPPER ENV IMPORT ERROR: {}\n".format(e))
    raise ImportError("Error from gripper_env.py: {}".format(e))
"""
Goal: Move gripper on top of the block (don't care about y coordinate)
State: Coordinate of Gripper, Coordinate of Block
Actions: Up/left, Right/Left
Success: Gripper directly above the block
Fail: Over 100 iterations (?)
"""
class GripperEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.step_count = 0
        self.max_steps = 100
        
        # load xml model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "GripperGPT.xml")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # actuator ids
        self.updown = self.model.actuator("up/down").id
        self.leftright = self.model.actuator("left/right").id
        self.forwardback = self.model.actuator("forward/backward").id
        self.actuators = np.array([self.updown, self.leftright, self.forwardback], dtype=int)
        
        # body ids
        self.body = self.model.body("block").id
        self.target = self.model.body("target").id
        self.gripper = self.model.body("gripper").id

        # action = 3 motors, obs = [block_xy, gripper_xy]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # in __init__
        self.ctrl_scale = np.array([15.0, 10.0, 10.0], dtype=float)  # [up/down, left/right, forward/back]

        # Define supported render modes
        self.render_modes = ["rgb_array"]

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1, 1)
        action *= self.ctrl_scale
        self.data.ctrl[self.updown] = action[0]
        self.data.ctrl[self.leftright] = action[1]
        self.data.ctrl[self.forwardback] = action[2]

        # advance physics
        for i in range(1,10):
            mujoco.mj_step(self.model, self.data)

        block_xy   = self.data.xpos[self.body][:2]
        gripper_xy = self.data.xpos[self.gripper][:2]
        obs = np.concatenate([block_xy, gripper_xy])
        
        # distance between block and gripper
        dist = np.linalg.norm(gripper_xy - block_xy)

        reward = -dist
        terminated = dist <= 0.05   # success threshold
        if dist <= 0.05:
            reward += 1
        truncated = self.step_count >= self.max_steps
        info = {"distance": dist}

        return obs, reward, terminated, truncated, info

    def render(self, width=384, height=384):
        """Render the environment and return RGB frame for video recording."""
        # Get camera ID for the main camera
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "main_camera")
        
        # Render the scene
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=camera_id)
        pixels = renderer.render()
        
        return pixels

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        rand_spawn(self.model, self.data)  # randomize block/target
        mujoco.mj_forward(self.model, self.data)  # propagate physics

        block_xy   = self.data.xpos[self.body][:2]
        gripper_xy = self.data.xpos[self.gripper][:2]
        obs = np.concatenate([block_xy, gripper_xy])

        return obs, {}


# Register the environment with Gymnasium
gym.register(
    id='Gripper-v1',
    entry_point='gripper_env:GripperEnv',
    max_episode_steps=100,
)