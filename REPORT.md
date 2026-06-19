\# IE306 Term Project Report



\## Reinforcement Learning for City-Scale Drone Delivery



\### Team Members and Roles



| Member       | Main Responsibility                                                                                                                                |

| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |

| Fırat Bozkur | Environment setup, baseline evaluation, value-based RL experiments, improved greedy teacher policy, behavioral cloning pipeline, ablation analysis |

| Teammate 1   | Policy-based method / actor-critic experiments                                                                                                     |

| Teammate 2   | Planning-based method / offline RL / multi-agent component                                                                                         |



> Note: The role table should be updated according to the final contribution split before submission.



\---



\## 1. Problem Description



The project focuses on a city-scale drone dispatch problem. At each decision step, the policy observes the current drone states, active delivery orders, city grid, time information, and a valid action mask. The policy must decide whether to assign a drone to an order, send a drone to charge, or take no action.



The main objective is to minimize `cost\_per\_order`, which is the primary evaluation metric. Lower `cost\_per\_order` indicates a better dispatch policy. The project baseline to beat is `greedy\_nearest`.



The action space is discrete. In the standard evaluation configuration, the action space consists of drone-order assignment actions, drone charging actions, and one no-op action. Since not every action is valid at every state, all learned policies use the provided action mask during action selection.



\---



\## 2. Evaluation Setup



All reported results were evaluated on:



```text

configs/eval\_standard.yaml

evaluation seeds: 0, 1, 2

main metric: cost\_per\_order

```



The main comparison baselines were:



| Policy               | Description                                                  |

| -------------------- | ------------------------------------------------------------ |

| `random`             | Random valid action baseline                                 |

| `greedy\_nearest`     | Official greedy nearest-order baseline                       |

| `milp\_rolling`       | Rolling MILP-based baseline                                  |

| `improved\_greedy`    | Custom heuristic teacher policy developed during the project |

| `behavioral\_cloning` | Learned policy trained from improved greedy demonstrations   |



\---



\## 3. Baseline Results



| Method                                  | cost\_per\_order | n\_delivered | n\_dropped | depletion\_events | episode\_return |

| --------------------------------------- | -------------: | ----------: | --------: | ---------------: | -------------: |

| Random                                  |         18.780 |       39.67 |     21.67 |             8.00 |        -168.33 |

| Greedy Nearest                          |          4.570 |      118.33 |     20.00 |             4.00 |        1183.26 |

| MILP Rolling                            |          4.722 |      118.00 |     23.00 |             3.33 |        1173.00 |

| Improved Greedy Teacher, threshold 0.55 |          1.388 |      133.33 |      9.67 |             0.00 |        1778.29 |



The official `greedy\_nearest` baseline achieved a `cost\_per\_order` of 4.570. The rolling MILP baseline was slightly worse, with a `cost\_per\_order` of 4.722. The custom improved greedy policy achieved the best overall heuristic performance with a `cost\_per\_order` of 1.388.



The improved greedy policy is not considered a learned RL method. Instead, it was used as a strong teacher policy for behavioral cloning.



\---



\## 4. Value-Based RL Methods



\### 4.1 DQN



A DQN agent was implemented using a feed-forward neural network that maps the flattened observation vector to Q-values for all discrete actions. Invalid actions were masked before action selection.



The initial DQN experiments showed unstable learning and poor performance. Several variants were tested:



| DQN Variant                    | cost\_per\_order |

| ------------------------------ | -------------: |

| Naive DQN                      |          28.93 |

| Assignment-priority DQN        |          16.36 |

| Battery-aware DQN              |          19.88 |

| Normalized + reward-scaled DQN |          20.27 |

| Greedy-guided DQN              |          24.43 |



The best DQN variant was the assignment-priority DQN with a `cost\_per\_order` of 16.36. However, it did not outperform the `greedy\_nearest` baseline.



\### 4.2 Double DQN



Double DQN was implemented to reduce overestimation bias. In standard DQN, the target network both selects and evaluates the next action. In Double DQN, the online network selects the next action while the target network evaluates it.



The Double DQN result was:



| Method     | cost\_per\_order | n\_delivered | n\_dropped | depletion\_events |

| ---------- | -------------: | ----------: | --------: | ---------------: |

| Double DQN |         16.898 |       25.33 |      1.33 |             8.00 |



Double DQN did not improve over the best DQN variant in this environment.



\### 4.3 Dueling DQN



Dueling DQN was implemented by separating the value stream and advantage stream:



```text

Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

```



The Dueling DQN result was:



| Method      | cost\_per\_order | n\_delivered | n\_dropped | depletion\_events |

| ----------- | -------------: | ----------: | --------: | ---------------: |

| Dueling DQN |         18.707 |       23.67 |      2.33 |             8.00 |



Dueling DQN also failed to outperform the greedy baseline.



\---



\## 5. Diagnosis of DQN Failure



The value-based RL methods were functional but did not reach competitive performance. The main observed issues were:



1\. \*\*Large action space:\*\* The policy must choose among many drone-order combinations, charging actions, and no-op.

2\. \*\*Sparse and delayed consequences:\*\* A poor assignment may only affect cost after several future steps.

3\. \*\*Exploration quality:\*\* Random exploration often creates poor dispatch trajectories, which fills the replay buffer with low-quality transitions.

4\. \*\*Charging behavior:\*\* Some DQN variants failed to charge drones properly, causing high depletion events.

5\. \*\*Action masking sensitivity:\*\* Small changes in action masking had large effects on performance.



The best DQN-based result remained far above the greedy baseline. Therefore, the project direction shifted toward imitation learning using a stronger teacher policy.



\---



\## 6. Improved Greedy Teacher Policy



The official `greedy\_nearest` policy mainly focuses on assigning drones to nearby pickup locations. A custom improved greedy policy was developed with two key changes:



1\. Use a higher charging threshold.

2\. Score each assignment by total route distance:



```text

drone\_to\_pickup\_distance + pickup\_to\_dropoff\_distance

```



A threshold sweep was performed:



| Charge Threshold | cost\_per\_order | n\_delivered | n\_dropped | depletion\_events |

| ---------------: | -------------: | ----------: | --------: | ---------------: |

|             0.40 |          2.322 |      128.33 |     14.00 |             1.00 |

|             0.45 |          1.858 |      130.00 |     11.33 |             0.67 |

|             0.50 |          1.726 |      131.67 |     11.33 |             0.33 |

|             0.55 |          1.388 |      133.33 |      9.67 |             0.00 |

|             0.60 |          1.521 |      132.00 |     10.67 |             0.00 |



The best threshold was 0.55. This improved greedy teacher achieved:



```text

cost\_per\_order = 1.388

```



This is significantly better than the official `greedy\_nearest` baseline of 4.570.



\---



\## 7. Behavioral Cloning



Behavioral cloning was used to train a learned policy from the improved greedy teacher. The process was:



1\. Run the improved greedy teacher in the environment.

2\. Save `(state, action, action\_mask)` samples.

3\. Train a neural network policy using supervised learning.

4\. Evaluate the learned policy using the same simulator interface.



The dataset for seed 0 contained:



```text

Number of samples: 36,911

State dimension: 581

Number of actions: 169

Mean teacher return: 1726.89

Mean delivered: 130.53

Mean dropped: 10.26

```



The BC model was trained using cross-entropy loss over the teacher actions. The best validation accuracy for seed 0 was approximately 0.4795.



\---



\## 8. Behavioral Cloning Results



Three independent BC policies were trained using different training seeds. All were evaluated on seeds 0, 1, and 2 with teacher-style masking and evaluation charge threshold 0.56.



| Training Seed | cost\_per\_order | n\_delivered | n\_dropped | depletion\_events | episode\_return |

| ------------: | -------------: | ----------: | --------: | ---------------: | -------------: |

|        Seed 0 |          4.039 |      112.67 |     28.00 |             0.00 |        1217.76 |

|        Seed 1 |          4.045 |      114.00 |     27.33 |             0.33 |        1232.38 |

|        Seed 2 |          4.097 |      113.33 |     28.67 |             0.00 |        1220.18 |



Aggregate result:



```text

BC mean cost\_per\_order = 4.061

BC std cost\_per\_order = 0.032

Greedy nearest cost\_per\_order = 4.570

```



Therefore, the behavioral cloning policy outperformed the official `greedy\_nearest` baseline.



\---



\## 9. Ablation Study: Action Masking



An ablation study was performed using the same trained BC model under two different action masking settings:



| Setting                   | cost\_per\_order | n\_delivered | n\_dropped | depletion\_events | Beats Greedy? |

| ------------------------- | -------------: | ----------: | --------: | ---------------: | ------------- |

| Raw simulator action mask |          5.247 |      107.00 |     32.33 |             0.67 | No            |

| Teacher-style action mask |          4.039 |      112.67 |     28.00 |             0.00 | Yes           |



The teacher-style mask reduced `cost\_per\_order` by approximately 23.0% compared to the raw mask.



This ablation shows that structured action masking is a key component of the final learned policy. The neural network alone learned useful behavior, but teacher-style masking was necessary to convert it into a baseline-beating dispatch policy.



\---



\## 10. Final Result Summary



| Method                  | Type                     | cost\_per\_order | Beats Greedy Nearest? |

| ----------------------- | ------------------------ | -------------: | --------------------- |

| Random                  | Baseline                 |         18.780 | No                    |

| Greedy Nearest          | Official baseline        |          4.570 | Reference             |

| MILP Rolling            | Baseline                 |          4.722 | No                    |

| Best DQN Variant        | Learned RL               |         16.356 | No                    |

| Double DQN              | Learned RL               |         16.898 | No                    |

| Dueling DQN             | Learned RL               |         18.707 | No                    |

| Improved Greedy Teacher | Custom heuristic         |          1.388 | Yes                   |

| Behavioral Cloning      | Learned imitation policy |  4.061 ± 0.032 | Yes                   |



The best learned method was behavioral cloning from the improved greedy teacher. The best overall policy was the improved greedy teacher itself, but since it is a heuristic, the main learned-policy result is the BC policy.



\---



\## 11. Engineering Log: What Broke and How It Was Diagnosed



Several issues occurred during development:



1\. \*\*Naive DQN failed to learn competitive behavior.\*\*

&#x20;  The first DQN implementation produced very high `cost\_per\_order`. This indicated that the basic model-free RL setup was not sufficient for the environment.



2\. \*\*Action selection produced poor dispatch behavior.\*\*

&#x20;  The model sometimes selected inefficient valid actions. Assignment-priority masking improved DQN from approximately 28.93 to 16.36.



3\. \*\*Battery-aware masking reduced depletion but increased dropped orders.\*\*

&#x20;  This showed that charging logic alone was not enough. The agent became safer but less productive.



4\. \*\*Greedy-guided DQN did not improve performance.\*\*

&#x20;  Using the official greedy baseline as an exploration guide worsened the result. The replay buffer still did not produce a strong learned policy.



5\. \*\*Custom improved greedy policy revealed that the official greedy baseline was improvable.\*\*

&#x20;  Considering total route distance and increasing the charge threshold significantly improved performance.



6\. \*\*Behavioral cloning nearly matched the baseline with raw masking but did not beat it.\*\*

&#x20;  Adding teacher-style masking during evaluation allowed the learned BC policy to outperform `greedy\_nearest`.



\---



\## 12. AI Use Statement



AI assistance was used for:



\* Understanding the simulator structure and project requirements.

\* Planning the implementation workflow.

\* Debugging Git, Python, and training issues.

\* Drafting code structure for DQN, Double DQN, Dueling DQN, improved greedy, and behavioral cloning.

\* Interpreting experiment results.

\* Drafting report sections.



All code was executed locally, tested, and modified during the project workflow. Final responsibility for the submitted work belongs to the team.



\---



\## 13. Conclusion



The project started with standard value-based reinforcement learning methods: DQN, Double DQN, and Dueling DQN. These methods were implemented and trained successfully, but they did not outperform the `greedy\_nearest` baseline.



A custom improved greedy teacher policy was then developed. This policy achieved a much lower `cost\_per\_order` by considering total route distance and using a better charging threshold. This teacher was used to generate demonstration data for behavioral cloning.



The final learned behavioral cloning policy achieved:



```text

cost\_per\_order = 4.061 ± 0.032

```



This outperformed the official `greedy\_nearest` baseline:



```text

greedy\_nearest cost\_per\_order = 4.570

```



The ablation study showed that teacher-style action masking was essential for reaching this result. Overall, the final learned policy successfully beat the main baseline while the DQN family provided useful negative results and diagnostic insight.



