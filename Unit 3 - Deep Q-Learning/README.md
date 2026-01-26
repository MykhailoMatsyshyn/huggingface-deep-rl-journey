# 🎮 Deep Q-Learning: Space Invaders 🚀

This repository contains my solution for **Unit 3** of the [Hugging Face Deep Reinforcement Learning Course](https://huggingface.co/deep-rl-course/unit3/introduction).

I trained a **Deep Q-Network (DQN)** agent to master **Space Invaders**, one of the classic Atari games. This project demonstrates the transition from **tabular Q-Learning** to **deep reinforcement learning**, using neural networks to handle high-dimensional visual observations.

<br>

## 📚 What this unit explores

In this unit I explored the fundamental concepts that bridge **tabular methods** to **deep reinforcement learning**:

- **Tabular vs. Deep Q-Learning** – While tabular Q-Learning uses a table to represent Q-values for discrete state-action pairs, **Deep Q-Learning** uses a neural network to approximate Q-values when the state space is too large (like pixel observations from Atari games).
- **Temporal Limitation** – A single frame doesn't provide temporal information. To capture motion and dynamics, we **stack multiple frames together** (typically 4 frames) to give the network temporal context.
- **Experience Replay** – Instead of learning from consecutive experiences (which are highly correlated), we store experiences in a **replay buffer** and sample random batches. This breaks correlation, stabilizes training, and allows the agent to learn from the same experiences multiple times.
- **Fixed Q-Target** – To stabilize training, we use a **separate target network** with fixed parameters to calculate Q-targets. This prevents the target from moving every time we update the main network, reducing instability.
- **Double DQN** – A technique to handle **overestimation of Q-values** by using two networks: one to select the best action, and another (target network) to evaluate that action's value. This decouples action selection from value estimation.

These techniques are essential for scaling Q-Learning to complex visual environments like Atari games.

<br>

---

## 🧠 Understanding Deep Q-Learning

### Architecture Overview

Deep Q-Learning extends tabular Q-Learning by using a **Convolutional Neural Network (CNN)** to process raw pixel observations and approximate Q-values for each possible action.

<div align="center">

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/deep-q-network.jpg" alt="Deep Q-Network Architecture" width="80%"/>

_Deep Q-Network (DQN) Architecture_

</div>

**Key Components:**

- **Input**: Stacked frames (4 frames of 84x84 grayscale images)
- **Convolutional Layers**: Extract spatial features from the visual input
- **Fully Connected Layers**: Map features to Q-values for each action
- **Output**: Q-values for all possible actions in the action space

### Preprocessing and Temporal Information

<div align="center">

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/preprocessing.jpg" alt="Preprocessing Pipeline" width="80%"/>

_Preprocessing Pipeline: Frame Stacking for Temporal Information_

</div>

**Preprocessing Steps:**

1. **Resize to 84x84**: Reduce computational complexity while preserving essential game information
2. **Grayscale Conversion**: Convert RGB (3 channels) to grayscale (1 channel) since color isn't critical for Atari games
3. **Frame Stacking**: Stack 4 consecutive frames together to provide temporal context (motion, velocity, direction)

**Why Frame Stacking?**

- A single frame is **stateless** – it doesn't show movement or direction
- Stacking frames allows the network to see **temporal patterns** (e.g., a bullet moving, an enemy approaching)
- This is essential for making informed decisions in dynamic environments

### Training Algorithm: Sampling and Training Phases

<div align="center">

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit4/sampling-training.jpg" alt="Deep Q-Learning Training Algorithm" width="80%"/>

_Deep Q-Learning Training Algorithm: Two-Phase Process_

</div>

The Deep Q-Learning training process consists of **two distinct phases**:

#### 1. **Sampling Phase**

- Agent performs actions in the environment using an **epsilon-greedy policy**
- Observed experience tuples `(state, action, reward, next_state, done)` are stored in a **replay buffer**
- This phase collects diverse experiences for later learning

#### 2. **Training Phase**

- Random batches of experiences are sampled from the replay buffer
- The neural network updates its weights using **gradient descent** on the Bellman equation
- **Fixed Q-Target**: A separate target network (updated periodically) is used to calculate stable Q-targets

**Key Benefits:**

- **Experience Replay**: Breaks correlation between consecutive samples, prevents catastrophic forgetting
- **Random Sampling**: Reduces variance and stabilizes learning
- **Fixed Targets**: Prevents the target from moving during updates, leading to more stable convergence

<br>

---

## 📊 Project Overview

### 🛠️ Tech Stack & Libraries

- 🐍 **Python** – Core programming language
- 🧠 **PyTorch** – Deep learning framework for neural network implementation
- 🤖 **Stable-Baselines3** – High-level RL library with DQN implementation
- 🏋️ **[Gymnasium](https://gymnasium.farama.org/)** – Standard API for RL environments
- 🎮 **Atari Environments** – Classic arcade games for RL benchmarking
- 🤗 **Hugging Face Hub** – Model sharing and versioning

### 🎮 Environment

| Environment           | Description                                                              | Observation Space                      | Action Space       | Link                                                                             |
| :-------------------- | :----------------------------------------------------------------------- | :------------------------------------- | :----------------- | :------------------------------------------------------------------------------- |
| **Space Invaders** 🚀 | Classic arcade game: destroy alien invaders while avoiding their bullets | 210x160x3 RGB → 84x84x4 (preprocessed) | 6 discrete actions | [Documentation](https://gymnasium.farama.org/environments/atari/space_invaders/) |

**Action Space:**

- 0: NOOP (no operation)
- 1: FIRE
- 2: UP
- 3: RIGHT
- 4: LEFT
- 5: DOWN

<br>

---

## 🎯 Implementation Details

### ⚙️ Hyperparameters

- **Algorithm**: DQN (Deep Q-Network)
- **Training Timesteps**: 1,000,000+
- **Learning Rate**: 0.0001
- **Batch Size**: 32
- **Replay Buffer Size**: 100,000
- **Target Network Update Frequency**: Every 1,000 steps
- **Epsilon Decay**: Linear from 1.0 → 0.01
- **Discount Factor (γ)**: 0.99
- **Frame Stacking**: 4 frames

### 🛡️ Stabilization Techniques Used

1. **Experience Replay**
   - Stores 100,000 experiences in a replay buffer
   - Randomly samples batches to break correlation
   - Allows learning from past experiences multiple times

2. **Fixed Q-Target**
   - Separate target network with frozen parameters
   - Updated every 1,000 steps by copying main network weights
   - Prevents target values from moving during training

3. **Frame Preprocessing**
   - Resize to 84x84 pixels
   - Grayscale conversion (RGB → single channel)
   - Frame stacking (4 frames) for temporal information

<br>

---

## 📈 Results

<div align="center">

| Metric          | Value                         | Description                                  |
| :-------------- | :---------------------------- | :------------------------------------------- |
| **Environment** | `SpaceInvadersNoFrameskip-v4` | Atari Space Invaders environment             |
| **Algorithm**   | DQN (Deep Q-Network)          | Neural network-based Q-Learning              |
| **Status**      | ✅ **Trained**                | Successfully trained agent on Space Invaders |

_Model Hub:_ [MykhailoMatsyshyn/dqn-SpaceInvadersNoFrameskip-v4](https://huggingface.co/MykhailoMatsyshyn/dqn-SpaceInvadersNoFrameskip-v4)

</div>

### 🎥 Agent Demonstrations

Watch the trained DQN agent in action on different Atari environments:

#### Space Invaders

<div align="center">
  <video src="https://github.com/user-attachments/assets/1c2c95d5-847f-4222-83fa-642dee30b86c" controls="controls" style="max-width: 100%;">
  </video>
</div>

_Final trained model playing Space Invaders (steps 0-2000)_

#### Beam Rider

<div align="center">
  <video src="https://github.com/user-attachments/assets/2e44964c-dae1-4317-a8f0-fda5c2c9edc5" controls="controls" style="max-width: 100%;">
  </video>
</div>

_Final trained model playing Beam Rider (steps 0-2000)_

**Key Observations:**

- The agent learns to **navigate** and **shoot** at enemies
- It demonstrates **temporal understanding** through frame stacking (recognizing moving objects)
- The agent shows **strategic behavior** (positioning, timing shots)

<br>

---

## 🔧 Key Technical Challenges & Solutions

### 1. **High-Dimensional State Space**

**Challenge:** Atari games have 210x160x3 RGB observations (100,800 values per frame), making tabular methods impossible.

**Solution:**

- Preprocess frames to 84x84 grayscale (7,056 values per frame)
- Use CNN to extract meaningful features automatically
- Stack 4 frames for temporal context (28,224 total values)

### 2. **Temporal Information**

**Challenge:** A single frame doesn't show movement or direction (e.g., can't tell if a bullet is moving up or down).

**Solution:**

- Stack 4 consecutive frames together
- Network can now see motion patterns and make informed decisions

### 3. **Training Instability**

**Challenge:** Learning from consecutive experiences causes high correlation and unstable updates.

**Solution:**

- **Experience Replay**: Store experiences in a buffer, sample random batches
- **Fixed Q-Target**: Use a separate target network updated periodically
- This breaks correlation and stabilizes learning

### 4. **Overestimation of Q-Values**

**Challenge:** Using the same network to select and evaluate actions leads to overestimated Q-values.

**Solution:**

- **Double DQN**: Use main network to select action, target network to evaluate it
- Decouples action selection from value estimation
- Reduces overestimation bias

<br>

---

## 📚 Personal Takeaways from Unit 3

This unit was a **major leap** from tabular methods to deep learning. The most eye-opening realization was how **preprocessing and architecture choices** can make or break a deep RL agent.

**Frame stacking** felt like giving the agent a "memory" of recent frames - without it, the network is essentially blind to motion. **Experience replay** transformed training from chaotic to stable - learning from random batches instead of consecutive experiences made a huge difference.

The **fixed Q-target** technique was particularly elegant: by freezing the target network and updating it periodically, we prevent the "moving target" problem that plagues naive implementations. This simple trick dramatically improves convergence.

Working with **visual observations** (pixels) instead of discrete states made the problem feel much more "real" - this is how RL agents perceive the world in many practical applications. The CNN architecture automatically learned to extract relevant features (enemies, bullets, player position) without explicit feature engineering.

Finally, seeing the agent **actually play the game** after training was incredibly satisfying. From random actions to strategic gameplay - that's the magic of deep reinforcement learning.

<br>

---

## 🔗 References & Resources

- [Unit 3: Deep Q-Learning with Atari Games](https://huggingface.co/deep-rl-course/unit3/introduction)
- [Gymnasium Atari Documentation](https://gymnasium.farama.org/environments/atari/)
- [Stable-Baselines3 DQN Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- [Hugging Face Deep RL Course](https://huggingface.co/deep-rl-course)

### 📦 Model Repository

- [dqn-SpaceInvadersNoFrameskip-v4](https://huggingface.co/MykhailoMatsyshyn/dqn-SpaceInvadersNoFrameskip-v4)
