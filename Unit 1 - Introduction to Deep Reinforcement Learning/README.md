# 🚀 Deep RL: Lunar Lander with PPO 🌝

This repository contains my solution for **Unit 1** of the [Hugging Face Deep Reinforcement Learning Course](https://huggingface.co/deep-rl-course/unit1/introduction).

I trained a **PPO (Proximal Policy Optimization)** agent to autonomously land a spacecraft on the moon using the `Stable-Baselines3` library and `Gymnasium`.

<br>

## 📊 Evaluation Results

After training for 1,000,000 timesteps, the agent was evaluated over 10 episodes in a deterministic mode.

<div align="center">

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Mean Reward** | **263.21** +/- 22.74 | High score indicating precise landing. |
| **Environment** | `LunarLander-v3` | Validated on the updated Gymnasium environment. |
| **Status** | ✅ Solved | Exceeds the passing threshold of 200. |

*Model Hub:* [MykhailoMatsyshyn/ppo-LunarLander-v3](https://huggingface.co/MykhailoMatsyshyn/ppo-LunarLander-v3)
</div>


<br>

## 🎥 Agent Behavior (Video)

Below is a compilation of the agent landing on **10 different random terrains**, demonstrating its generalization capabilities.

<div align="center">
  <video src="https://github.com/user-attachments/assets/17ef44fd-138a-4afb-948c-442abaf5d7fc" controls="controls" style="max-width: 100%;">
  </video>
</div>

<br>

## ⚙️ Technical Implementation

### 1. Model Architecture & Hyperparameters

I used a Multilayer Perceptron (MLP) policy because the observation space is a vector (coordinates, speed, angle).

<div align="center">

| Hyperparameter | Value | Reason |
| :--- | :--- | :--- |
| `policy` | `MlpPolicy` | Input is a state vector, not an image. |
| `n_steps` | `1024` | Number of steps to run for each environment per update. |
| `batch_size` | `64` | Minibatch size for optimization. |
| `gamma` | `0.999` | High discount factor to prioritize safe landing (long-term goal). |
| `ent_coef` | `0.01` | Encourages exploration to prevent early local optima. |
| `gae_lambda` | `0.98` | Factor for trade-off of bias vs variance for GAE. |
| `n_epochs` | `4` | Number of optimization epochs. |

</div>


### 2. Overcoming Engineering Challenges

During development, I addressed several compatibility and MLOps challenges:

* **Gymnasium vs. Gym:** The course materials used the legacy `gym` library. I adapted the code to work with the modern `gymnasium` standard.
* **Library Compatibility (Shimmy):** To evaluate older models from the leaderboard, I integrated the [Shimmy](https://github.com/Farama-Foundation/Shimmy) API conversion tool to bridge the gap between `gym` and `gymnasium`.
* **Custom Video Pipeline:** The standard `package_to_hub` function had issues rendering video in the new environment versions. I implemented a custom pipeline that:
    1.  Records multiple episodes using `RecordVideo` wrapper.
    2.  Merges them into a single montage using `FFmpeg`.
    3.  Uploads the composite video to the Hub manually via `HfApi`.

<br>

## 🔗 References
* [Unit 1: Introduction to Deep Reinforcement Learning](https://huggingface.co/deep-rl-course/unit1/introduction)
* [Stable-Baselines3: Official Documentation](https://stable-baselines3.readthedocs.io/en/master/)
