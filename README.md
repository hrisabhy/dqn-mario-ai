# DQN Super Mario Bros

## Project Description
This project implements a Deep Q-Network (DQN) to play Super Mario Bros using reinforcement learning. The agent learns to play the game by interacting with the environment, receiving rewards, and using experience replay for training. It uses OpenAI's Gym and the Gym Super Mario Bros environment. The DQN architecture is based on convolutional layers to process game frames and learn optimal actions.

## Setup Instructions

### Prerequisites
Make sure you have Python 3.6+ installed

### Installing Dependencies

1. Clone the repository:

   ```bash
   git clone https://github.com/hrisabhy/dqn-mario-ai.git
   cd dqn-mario-ai
   ```

2. Create a virtual environment (recommended)
   ```
      python3 -m venv venv
      source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install the required Python packages
   ```
      pip install -r requirements.txt
   ```

### Running the Project
```
   python main.py
```