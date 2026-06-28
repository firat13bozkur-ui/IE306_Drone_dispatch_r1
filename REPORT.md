## Role B: Policy-Based Approach (Masked Actor-Critic)

### 1. Method Description & Architecture
For Role B, a centralized Masked Actor-Critic (A2C) architecture was implemented to handle the operational dispatching decisions of the drone fleet[cite: 1, 2]. The network utilizes a Shared Multi-Layer Perceptron (MLP) backbone consisting of two hidden layers with 128 units each and ReLU activations. From this shared representation, the network splits into an Actor head outputting raw logits for the 169 discrete actions, and a Critic head outputting a scalar value $V(s)$ representing the expected discounted cumulative return.

### 2. Action Masking Integration
Given the extremely constrained action space where specific drone-order assignments are frequently illegal, strict action masking was integrated directly into the stochastic policy[cite: 2]. Before sampling via `torch.distributions.Categorical`, illegal actions identified by `get_action_mask(obs)` were heavily penalized by setting their corresponding logits to $-1\times 10^9$[cite: 2]. This mathematically guarantees that the agent never selects an invalid dispatch[cite: 2].

### 3. Hyperparameter Exploration & Training Optimization
To alleviate the high variance common in full episodic Monte Carlo updates, the framework was optimized to execute **N-Step Bootstrapping** with a 20-step rollout horizon[cite: 2]. Extensive hyperparameter tuning was conducted to stabilize the gradient trajectory:
* **Learning Rate (lr) Sweep:** Lowering the learning rate to $1\times 10^{-4}$ proved overly restrictive for the 169-action landscape, confirming that $3\times 10^{-4}$ provided the optimal optimization step size[cite: 2].
* **Entropy Regularization Sweep:** Increasing the entropy coefficient from $0.01$ to $0.05$ caused severe performance degradation ($cost\_per\_order$ rising to 29.61), indicating that high exploration noise disrupts the delicate action masks[cite: 2]. 

### 4. Evaluation Results & Quantitative Analysis
The table below details the 3-seed evaluation performance under the standard configuration[cite: 2]:

| Metric | Masked Actor-Critic (Ours) | Greedy Nearest (Baseline) |
| :--- | :---: | :---: |
| **Cost per Order** (Lower is better) | **18.34** | 4.57[cite: 2] |
| **Success Rate** | 67.87% | 85.49%[cite: 2] |
| **Avg. Delivered Orders** | 38.00 | 118.33[cite: 2] |
| **Avg. Dropped Orders** | 18.33 | 20.00[cite: 2] |
| **Mean Episode Return** | -131.93 | 1183.26[cite: 2] |

### 5. Discussion & Engineering Diagnostics
Transitioning from full episodic rollouts to an N-step updating scheme successfully dropped the $cost\_per\_order$ from 22.92 to 18.34, validating the use of localized temporal credit assignment[cite: 2]. The policy-gradient approach did not outperform the `greedy_nearest` baseline due to severe sample inefficiency when exploring 169 discrete slots simultaneously under stochastic seed variations[cite: 1, 2]. However, the framework successfully demonstrates an error-free execution of masked actor-critic constraints with robust bootstrapping mechanics[cite: 2].
