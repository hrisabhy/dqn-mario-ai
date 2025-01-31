import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
from collections import deque
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros import make

# --- 1. PREPROCESSING FUNCTION ---
# Preprocesses input frames: converts to grayscale, resizes, and normalizes
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (84, 84))  # Resize to 84x84
    frame = frame.astype(np.float32) / 255.0  # Normalize pixel values
    return frame

# --- 2. CREATE SUPER MARIO ENVIRONMENT ---
# Initializes the Mario environment with simplified action space
env = make("SuperMarioBros-1-1-v0")  # Load Mario
env = JoypadSpace(env, SIMPLE_MOVEMENT)  # Reduce action space to basic movements
env.metadata['video.frames_per_second'] = 120  # Set FPS for rendering

# --- 3. DEFINE DQN NETWORK ---
# Defines the neural network model for the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),  # Convolutional layers
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),  # Flatten output for dense layers
            nn.Linear(64 * 7 * 7, 512),  # Fully connected layers
            nn.ReLU(),
            nn.Linear(512, output_dim)  # Output Q-values for each action
        )

    def forward(self, x):
        return self.network(x)

# --- 4. TRAINING PARAMETERS ---
# Setting hyperparameters and other constants for the DQN training process
GAMMA = 0.99  # Discount factor for future rewards
LEARNING_RATE = 0.00025  # Learning rate for optimizer
MEMORY_SIZE = 10000  # Max size of replay memory
BATCH_SIZE = 64  # Batch size for training
EPSILON_START = 1.0  # Initial epsilon for epsilon-greedy strategy
EPSILON_MIN = 0.1  # Minimum epsilon
EPSILON_DECAY = 0.995  # Decay rate for epsilon
TARGET_UPDATE = 10  # Number of episodes before updating target network
NUM_EPISODES = 500  # Number of training episodes

# --- 5. INITIALIZE DQN ---
# Initialize the policy and target networks, and the optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
state_shape = (4, 84, 84)  # Stack of 4 frames as input
n_actions = env.action_space.n  # Number of possible actions
policy_net = DQN(input_dim=4, output_dim=n_actions).to(device)  # Policy network
target_net = DQN(input_dim=4, output_dim=n_actions).to(device)  # Target network
target_net.load_state_dict(policy_net.state_dict())  # Copy weights from policy network to target network
target_net.eval()  # Set target network to evaluation mode
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)  # Adam optimizer
memory = deque(maxlen=MEMORY_SIZE)  # Replay memory

# --- 6. EPSILON-GREEDY STRATEGY ---
# Chooses action based on epsilon-greedy strategy (explore vs exploit)
def select_action(state, epsilon):
    if random.random() < epsilon:  # Exploration: choose random action
        return random.randint(0, n_actions - 1)
    else:  # Exploitation: choose best action from Q-values
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return policy_net(state_tensor).argmax().item()  # Action with highest Q-value

# --- 7. EXPERIENCE REPLAY FUNCTION ---
# Updates the DQN using mini-batch of experiences sampled from memory
def optimize_model():
    if len(memory) < BATCH_SIZE:  # If memory is too small, skip optimization
        return

    batch = random.sample(memory, BATCH_SIZE)  # Sample random batch from memory
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)  # Convert to tensor
    actions = torch.tensor(np.array(actions), dtype=torch.int64, device=device).unsqueeze(1)  # Actions tensor
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)  # Rewards tensor
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)  # Next states tensor
    dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device)  # Done flags tensor

    current_q_values = policy_net(states).gather(1, actions).squeeze()  # Get Q-values for current states
    next_q_values = target_net(next_states).max(1)[0].detach()  # Get max Q-value for next states
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))  # Calculate expected Q-values

    loss = nn.functional.mse_loss(current_q_values, expected_q_values)  # Calculate loss (MSE)
    optimizer.zero_grad()  # Zero out gradients before backpropagation
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the policy network's weights

# --- 8. TRAINING LOOP ---
# Main training loop for DQN
epsilon = EPSILON_START  # Start with high exploration
for episode in range(NUM_EPISODES):
    state = env.reset()  # Reset environment at the start of each episode
    state = preprocess_frame(state)  # Preprocess the initial frame
    state_stack = np.stack([state] * 4, axis=0)  # Stack 4 frames as input
    total_reward = 0

    while True:
        env.render()  # Render environment for visualization
        action = select_action(state_stack, epsilon)  # Select an action based on epsilon-greedy
        next_state, reward, done, _ = env.step(action)  # Take action in environment and observe result
        next_state = preprocess_frame(next_state)  # Preprocess the next state
        next_state_stack = np.roll(state_stack, shift=-1, axis=0)  # Update state stack (FIFO)
        next_state_stack[-1] = next_state  # Add new state to stack

        # Reward shaping: penalize if done (episode ended)
        if done:
            reward = -1

        memory.append((state_stack, action, reward, next_state_stack, done))  # Store experience in memory
        state_stack = next_state_stack  # Update state stack
        total_reward += reward

        optimize_model()  # Perform one optimization step

        if done:
            break  # End episode if done

    if epsilon > EPSILON_MIN:  # Decay epsilon to shift from exploration to exploitation
        epsilon *= EPSILON_DECAY

    if episode % TARGET_UPDATE == 0:  # Periodically update target network
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode + 1}/{NUM_EPISODES}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")  # Print episode stats

print("Training complete!")
env.close()  # Close the environment after training
