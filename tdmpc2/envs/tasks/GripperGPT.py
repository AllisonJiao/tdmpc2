import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("GripperGPT.xml")
data = mujoco.MjData(model)

# Convenience: indices
left_ctrl_id = 0
right_ctrl_id = 1
gripper_qpos_addr = model.joint("gripper_free").qposadr  # starting index in qpos

def set_gripper_pos(data, pos):
    """Set gripper position (x,y,z), keep orientation fixed (identity quaternion)."""
    addr = gripper_qpos_addr[0]
    data.qpos[addr:addr+3] = pos
    data.qpos[addr+3:addr+7] = np.array([1, 0, 0, 0])  # identity quaternion

# Initial placement
set_gripper_pos(data, np.array([0, 0, 0.3]))
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        t = time.time() - start

        # Scripted sequence
        if t < 1.0:
            # Hover above block
            set_gripper_pos(data, np.array([0, 0, 0.3]))
            data.ctrl[left_ctrl_id] = 0.04
            data.ctrl[right_ctrl_id] = -0.04
        elif t < 2.0:
            # Lower down
            set_gripper_pos(data, np.array([0, 0, 0.15]))
        elif t < 3.0:
            # Close fingers
            data.ctrl[left_ctrl_id] = 0.0
            data.ctrl[right_ctrl_id] = 0.0
        elif t < 4.0:
            # Lift
            set_gripper_pos(data, np.array([0, 0, 0.3]))
        elif t < 5.5:
            # Move sideways
            set_gripper_pos(data, np.array([0.3*(t-4.0), 0, 0.3]))
        elif t < 7.0:
            # Open fingers to drop
            data.ctrl[left_ctrl_id] = 0.04
            data.ctrl[right_ctrl_id] = -0.04
        else:
            # End: hover away
            set_gripper_pos(data, np.array([0.3, 0, 0.3]))

        # Step simulation
        mujoco.mj_step(model, data)

        # sync rendering at ~real-time
        viewer.sync()
