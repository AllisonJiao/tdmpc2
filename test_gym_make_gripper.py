#!/usr/bin/env python3
"""
Test script to verify that the gripper environment works with gym.make()
just like other gymnasium environments.
"""

import os
import sys
import numpy as np
import gymnasium as gym

# Add the project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'tdmpc2', 'envs', 'tasks'))

def test_gym_make_gripper():
    """Test creating gripper environment with gym.make()"""
    print("=== Testing gym.make() with Gripper Environment ===")
    
    try:
        # Import and register the environment
        from gripper_env import GripperEnv
        
        # Register the environment
        gym.register(
            id='Gripper-v1',
            entry_point='gripper_env:GripperEnv',
            max_episode_steps=100,
        )
        print("✓ Environment registered successfully")
        
        # Create environment using gym.make()
        env = gym.make('Gripper-v1', render_mode='rgb_array')
        print("✓ Environment created with gym.make()")
        
        # Test basic functionality
        obs, info = env.reset()
        print(f"✓ Environment reset successful, obs shape: {obs.shape}")
        
        # Test step
        action = np.array([0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful, reward: {reward}")
        
        # Test rendering
        frame = env.render()
        print(f"✓ Environment render successful, frame shape: {frame.shape}")
        
        # Test episode
        obs, info = env.reset()
        done = False
        step_count = 0
        while not done and step_count < 10:  # Short test episode
            action = np.random.uniform(-0.5, 0.5, size=3)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            # Test rendering during episode
            frame = env.render()
        
        print(f"✓ Episode completed successfully in {step_count} steps")
        
        print("\n=== All tests passed! ===")
        print("Your gripper environment now works with gym.make() just like other environments!")
        print("\nUsage examples:")
        print("  env = gym.make('Gripper-v1', render_mode='rgb_array')")
        print("  env = gym.make('Gripper-v1')  # without render mode")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gym_make_gripper()
    sys.exit(0 if success else 1)
