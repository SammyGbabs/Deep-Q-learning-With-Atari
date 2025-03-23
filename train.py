# import gymnasium as gym
# from stable_baselines3 import DQN
# from stable_baselines3.dqn.policies import CnnPolicy
# import os

# # Create Pong environment
# env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

# # Define the model using a Convolutional Neural Network (CNN) policy
# model = DQN(
#     policy=CnnPolicy,
#     env=env,
#     learning_rate=0.0001,  # Fine-tune for better performance
#     gamma=0.99,  # Discount factor
#     batch_size=32,  # Mini-batch size
#     exploration_initial_eps=1.0,  # Start with full exploration
#     exploration_final_eps=0.1,  # Minimum exploration
#     exploration_fraction=0.1,  # Fraction of training over which exploration decreases
#     verbose=1,
# )

# # Train the model
# TIMESTEPS = 500_000  # Adjust based on computational power
# model.learn(total_timesteps=TIMESTEPS)

# # Save the trained model
# model_path = "dqn_pong_model.zip"
# model.save(model_path)
# print(f"Model saved to {model_path}")

# # Close environment
# env.close()
#!/usr/bin/env python3
"""Training an agent"""


import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import ale_py

class BreakoutAgent:
    """Handling the setup, execution"""

    def __init__(self, model_directory="models", log_dir="logs", total_timesteps=50000):
        """Setting up the agent, including environment and model loading"""
        self.model_directory = model_directory
        self.log_dir = log_dir
        self.total_timesteps = total_timesteps

        # Creating the directories
        os.makedirs(self.model_directory, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initializing environments
        self.env = self._create_wrapped_env()
        self.eval_env = self._create_wrapped_env()

        # DQN model
        self.model = self._initialize_model()
        

    @staticmethod
    def _create_env(render_mode=None):
        """Creating the Breakout environment"""
        env = gym.make("ALE/Breakout-v5", render_mode=render_mode)
        env = Monitor(env)
        return env

    def _create_wrapped_env(self):
        """Vectorized environment"""
        return DummyVecEnv([lambda: self._create_env()])
    
    def _initialize_model(self):
        """Initializing the model"""
        return DQN(
            "CnnPolicy",
            self.env,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            verbose=1,
            tensorboard_log=self.log_dir,
        )

    def train(self):
        """Training the DQN agent"""
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, save_path=self.model_directory, name_prefix="dqn_breakout"
        )
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{self.model_directory}/best_model",
            log_path=self.log_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        # Training
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )

        # Saving the model
        self.model.save(f"{self.model_directory}/policy.zip")
        print(f"Training completed! Model saved as '{self.model_directory}/policy.zip'")


    # def load_model(self):
    #     """Loading the DQN agent's trained model"""
    #     try:
    #         loaded_model = DQN.load(self.model_file) #Loading model from .zip file
    #         print(f"Model loaded successfully from: {self.model_file}")
    #         return loaded_model
    #     except Exception as e:
    #         print(f"Failed to load model: {e}")
    #         raise RuntimeError("Check the model file")

    def execute(self, episodes=5):
        """Allowing the agent to play the game for 5 episodes"""
        for ep in range(episodes):
            observation = self.eval_env.reset()
            total_points = 0
            done = False
            # move_count = 0
            
            while not done:
                # Selecting the action using the pre-trained model
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, done, _ = self.eval_env.step(action)
                total_points += reward
                # move_count += 1
                # done = terminated or truncated

            print(
                f"Episode {ep + 1}: Total Score: {total_points}"
            )

        self.env.close()

    # def convert_to_h5(self, h5_filepath="models/agent_policy.h5"):
    #     """Converting the trained model (PyTorch) to an HDF5 (.h5) file for compatibility to avoid errors"""
    #     print("Initiating PyTorch to HDF5 model conversion...")

    #     # Accessing the PyTorch layers
    #     torch_model = self.agent_model.policy.q_net

    #     # Creating a Keras sequential model and converting each layer
    #     keras_model = tf.keras.Sequential()
    #     for layer in torch_model.children():
    #         if isinstance(layer, torch.nn.Conv2d):
    #             keras_model.add(
    #                 tf.keras.layers.Conv2D(
    #                     filters=layer.out_channels,
    #                     kernel_size=layer.kernel_size,
    #                     strides=layer.stride,
    #                     activation="relu",
    #                     padding="valid",
    #                     input_shape=(210, 160, 3),
    #                 )
    #             )
    #         elif isinstance(layer, torch.nn.Linear):
    #             keras_model.add(
    #                 tf.keras.layers.Dense(
    #                     units=layer.out_features,
    #                     activation="relu"
    #                     if layer.weight.shape[0] > layer.weight.shape[1]
    #                     else None,
    #                 )
    #             )
    #         elif isinstance(layer, torch.nn.Flatten):
    #             keras_model.add(tf.keras.layers.Flatten())

    #     # Saving the Keras model
    #     keras_model.save(h5_filepath)
    #     print(f"HDF5 model saved successfully at: {h5_filepath}")


def run_agent():
    """Initialization and gameplay of the Breakout agent"""
    gym.register_envs(ale_py)

    # Initialize the trainer
    trainer = BreakoutAgent(total_timesteps=50000)

    # Train the agent
    trainer.train()

    # Evaluate the agent
    trainer.evaluate(episodes=5)

if __name__ == "__main__":
    run_agent()