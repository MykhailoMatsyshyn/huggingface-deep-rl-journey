# 🚀 Hugging Face Deep Reinforcement Learning Journey

<div align="center">

<img src="https://github.com/user-attachments/assets/bb637506-f0f1-4c14-a2f3-397abc34605c" alt="Deep RL Course" width="100%"/>

**My complete journey through the [Hugging Face Deep Reinforcement Learning Course](https://huggingface.co/deep-rl-course)**

*Building intelligent agents from scratch using state-of-the-art RL algorithms*

</div>

<br>

---

## 📖 About This Repository

This repository contains my solutions and implementations for the **Hugging Face Deep Reinforcement Learning Course**. Each unit demonstrates practical applications of reinforcement learning algorithms, from basic Q-Learning to advanced policy optimization methods.

**Key Highlights:**
- 🎯 **Hands-on implementations** of RL algorithms from scratch
- 📊 **Comprehensive experiments** with hyperparameter tuning and analysis
- 🎬 **Video demonstrations** of trained agents in action
- 🤗 **Model sharing** via Hugging Face Hub
- 📈 **Detailed documentation** with insights and learnings

<br>

---

## 📊 Progress Overview

<div align="center">

| Unit | Status | Environment | Algorithm | Link |
| :--- | :--- | :--- | :--- | :--- |
| **Unit 1** | ✅ **Completed** | LunarLander-v2 | PPO | [View →](Unit%201%20-%20Introduction%20to%20Deep%20Reinforcement%20Learning/) |
| **Unit 2** | ✅ **Completed** | FrozenLake, Taxi-v3 | Q-Learning | [View →](Unit%202%20-%20Introduction%20to%20Q-Learning/) |
| **Unit 3** | ✅ **Completed** | Space Invaders | DQN | [View →](Unit%203%20-%20Deep%20Q-Learning/) |
| **Unit 4** | 🚧 In Progress | PixelCopter, CartPole | Policy Gradient | Coming Soon |
| **Unit 5** | 🔜 Coming Soon | SnowballTarget, Pyramids | MLAgents | - |
| **Unit 6** | 🔜 Coming Soon | Panda Gym, PyBullet | Actor-Critic | - |
| **Unit 7** | 🔜 Coming Soon | SoccerTwos | Multi-Agent | - |
| **Unit 8** | 🔜 Coming Soon | LunarLander-v2, VizDoom | PPO | - |

**Progress: 3/8 Core Units Completed (37.5%)**

</div>

<br>

---

## 🗺️ Course Structure

### **Part 1: Foundations**

#### ✅ [Unit 1: Introduction to Deep Reinforcement Learning](Unit%201%20-%20Introduction%20to%20Deep%20Reinforcement%20Learning/)
**Environment:** `LunarLander-v2`  
**Algorithm:** PPO (Proximal Policy Optimization)  
**Status:** ✅ **Completed**

*Trained a PPO agent to autonomously land a spacecraft on the moon using Stable-Baselines3. Achieved a mean reward of 263.21 ± 22.74, successfully solving the environment.*

---

#### ✅ [Unit 2: Introduction to Q-Learning](Unit%202%20-%20Introduction%20to%20Q-Learning/)
**Environments:** `FrozenLake-v1` (4x4, non-slippery & slippery), `Taxi-v3`  
**Algorithm:** Q-Learning (from scratch with Python & NumPy)  
**Status:** ✅ **Completed**

*Implemented Q-Learning from scratch to master two classic environments. Conducted comprehensive hyperparameter experiments comparing 5 different configurations on stochastic FrozenLake, achieving 91% win rate with the optimal setup.*

---

#### ✅ [Unit 3: Deep Q-Learning with Atari Games](Unit%203%20-%20Deep%20Q-Learning/)
**Environment:** `Space Invaders`  
**Algorithm:** Deep Q-Network (DQN)  
**Status:** ✅ **Completed**

*Trained a DQN agent to play Space Invaders using neural networks to handle high-dimensional visual observations. Implemented experience replay, fixed Q-targets, and frame stacking to stabilize training and capture temporal information.*

---

#### 🔜 Bonus Unit: Hyperparameter search with Optuna
**Status:** 🚧 **Coming Soon**

---

### **Part 2: Advanced Methods**

#### 🚧 Unit 4: Policy Gradient with PyTorch
**Environments:** `PixelCopter`, `CartPole`  
**Algorithm:** Policy Gradient Methods  
**Status:** 🚧 **In Progress**

---

#### 🔜 Unit 5: Introduction to MLAgents
**Environments:** `SnowballTarget`, `Pyramids`  
**Status:** 🚧 **Coming Soon**

---

#### 🔜 Unit 6: Actor-Critic Methods with Robotics environments
**Environments:** `Panda Gym`, `PyBullet`  
**Algorithm:** Actor-Critic Methods  
**Status:** 🚧 **Coming Soon**

---

#### 🔜 Unit 7: Introduction to Multi-Agents Challenge: AI vs. AI
**Environment:** `SoccerTwos`  
**Status:** 🚧 **Coming Soon**

---

#### 🔜 Unit 8: Proximal Policy Optimization (PPO)
**Environments:** `LunarLander-v2`, `VizDoom` (Health Gathering and Deathmatch)  
**Algorithm:** PPO (CleanRL & Sample Factory)  
**Status:** 🚧 **Coming Soon**

---

### **Part 3: Advanced Topics**

#### 🔜 Optional Unit: Advanced Topics in Reinforcement Learning
**Topics:** Decision Transformers, RLHF, Language Models in RL, MineRL  
**Status:** 🚧 **Coming Soon**

---

<br>

---

## 🎯 Key Achievements

### 🏆 Unit 1: Lunar Lander
- **Mean Reward:** 263.21 ± 22.74
- **Status:** ✅ Solved (exceeds 200 threshold)
- **Model Hub:** [ppo-LunarLander-v3](https://huggingface.co/MykhailoMatsyshyn/ppo-LunarLander-v3)

### 🏆 Unit 2: Q-Learning
- **FrozenLake (Non-Slippery):** 100% success rate
- **FrozenLake (Slippery):** 91% win rate (best configuration)
- **Taxi-v3:** Mean reward 7.52 ± 2.74
- **Model Hubs:** 
  - [q-FrozenLake-v1-4x4-noSlippery](https://huggingface.co/MykhailoMatsyshyn/q-FrozenLake-v1-4x4-noSlippery)
  - [q-Taxi-v3](https://huggingface.co/MykhailoMatsyshyn/q-Taxi-v3)

### 🏆 Unit 3: Deep Q-Learning
- **Environment:** Space Invaders (Atari)
- **Mean Reward:** 838.5 ± 373.08 (evaluation)
- **Status:** ✅ Trained successfully
- **Model Hub:** [dqn-SpaceInvadersNoFrameskip-v4](https://huggingface.co/MykhailoMatsyshyn/dqn-SpaceInvadersNoFrameskip-v4)

<br>

---

## 🛠️ Tech Stack

<div align="center">

| Technology | Usage |
| :--- | :--- |
| **Python** | Core programming language |
| **NumPy** | Matrix operations and Q-table management |
| **PyTorch** | Deep learning framework (for later units) |
| **Stable-Baselines3** | High-level RL algorithms (PPO) |
| **Gymnasium** | RL environments (LunarLander, FrozenLake, Taxi) |
| **Hugging Face Hub** | Model sharing and versioning |
| **Matplotlib & Pandas** | Visualization and analysis |
| **ImageIO** | Video generation for demonstrations |

</div>

<br>

---

## 📚 Learning Resources

- 📖 [Hugging Face Deep RL Course](https://huggingface.co/deep-rl-course)
- 📚 [Gymnasium Documentation](https://gymnasium.farama.org/)
- 🤗 [Hugging Face Hub](https://huggingface.co/)
- 📊 [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

<br>

---

## 🤝 Contributing

This is a personal learning journey repository. However, if you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request!

<br>

---

## 📝 License

This repository is for educational purposes as part of the Hugging Face Deep Reinforcement Learning Course.

<br>

---

<div align="center">

**Built with ❤️ as part of the Hugging Face Deep Reinforcement Learning Course**

</div>
