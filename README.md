# Deep Reinforcement Learning for Autonomous Sumo Robot Simulation
Simulation Preview : https://gyazo.com/7156d71cf136808a7245d68ea787ea06
## Abstract

This project explores the application of Deep Reinforcement Learning (DRL), specifically the Proximal Policy Optimization (PPO) algorithm, for training autonomous Sumo robots in simulated competitive environments. Using Gymnasium and PyBullet simulation platforms, the project successfully demonstrates enhanced robotic behavior and strategy development through iterative training and simulation experiments. The results highlight improvements in robot performance, adaptability, and overall effectiveness in simulated competitions.

## Chapter 1: Introduction

### 1.1 Background and Motivation

Autonomous robotic systems continue to evolve rapidly, with deep reinforcement learning (DRL) offering significant advantages in adaptability and decision-making under uncertainty. Sumo robotics, an engaging competitive domain, serves as an ideal testing ground due to clearly defined rules and measurable objectives, facilitating meaningful experimentation with DRL algorithms.

### 1.2 Problem Statement

Traditional robotic strategies in sumo competitions typically rely on heuristic-driven methods, which lack adaptability and responsiveness to dynamic environments. DRL presents a viable alternative by enabling robots to autonomously learn optimal strategies directly through interaction with the environment.

### 1.3 Objectives of the Project

* Develop comprehensive virtual simulation environments for autonomous sumo robots.
* Apply the PPO algorithm effectively to train robust sumo robot behaviors.
* Analyze and compare robot performance across multiple simulation configurations.

### 1.4 Scope of Work

This project concentrates exclusively on virtual simulations without hardware implementation. It emphasizes algorithmic efficiency and strategy robustness within simulated competitive scenarios.

Main scopes including
* Train two robots to compete using a single PPO agent
* Focus on continuous control and shared-reward shaping
* Compare behavior under four different reward strategies
* Evaluate performance through simulation-only (no real robot)


## Chapter 2: Literature Review

### OpenAI Five (OpenAI, 2019): 
- Demonstrated the effectiveness of self-play in training agents for complex multi-agent games. From this, we adopted the concept of using past versions of the agent as opponents to drive continuous improvement.

### Proximal Policy Optimization (Schulman et al., 2017): 
- Selected as our core RL algorithm due to its stable training and strong empirical performance across diverse tasks. We specifically use PPO for its clipped objective function, which helps maintain stable updates in adversarial settings.

### Multi-Agent Emergent Behavior (e.g., Competitive Environments in PettingZoo): 
- We referenced common self-play setups in open-source environments where agents learn competitive behavior (e.g., sumo, wrestling) through interaction. We adapted this idea to train our sumo agents by pairing current agents with older policy snapshots.

### 2. Sumo Robot Competitions

Sumo robot competitions involve autonomous robots attempting to push opponents out of a predefined arena. These competitions serve as effective platforms for evaluating autonomous navigation, strategy development, and robotic strength.

## Chapter 3: Methodology

### 3.1 Overview of the Methodology

The methodology integrates robot design, environment development, DRL training, and comprehensive analysis, ensuring a holistic approach to developing robust autonomous behaviors.

### 3.2 Sumo Robot Design

The robot is modeled using URDF (Unified Robot Description Format), defining realistic physical properties such as mass distribution, wheel configuration, and sensor placements crucial for accurate simulations.

### 3.3 Simulation Environment Setup

Four distinct simulation environments (`sumo_env1.py` to `sumo_env4.py`) were developed with incremental complexity and realism, utilizing Gymnasium and PyBullet to simulate realistic physical interactions and robot dynamics.

### 3.4 Deep Reinforcement Learning Approach

The PPO algorithm was selected due to its stability and suitability for continuous control tasks. Training scripts (`train1.py` to `train4.py`) include comprehensive hyperparameter tuning, ensuring optimal policy development.

### 3.5 Experimental Setup

The training experiments utilized systematic checkpointing, detailed logging (TensorBoard), and were conducted on computational resources designed to ensure efficient and repeatable experimentation.

## Chapter 4: Results and Discussion

### 4.1 Training Performance Analysis

Training performance was quantitatively assessed through episode reward curves. Comparisons across reward designs (Base Setting, More Winning Reward, More Penalty, Low Friction Floor) demonstrated clear variations in robot learning speed, stability, and aggressiveness.

### 4.2 Policy and Behavioral Analysis

Detailed behavioral analysis showed distinct strategies emerging from different reward structures:

* **Standard Observation:** Balanced and symmetric behaviors.
* **Biased Observation:** Distinct aggressive and defensive roles, despite equal reward signals.
* **Decay Over Time:** Enhanced early exploration, with conservative late-match strategies.
* **Slippery Floor Observation:** Conservative behaviors aimed at stability and avoiding boundary exits.

### 4.3 Comparative Analysis Across Environments

Performance across different environments highlighted how specific reward adjustments and environmental friction significantly impacted strategic behavior, learning effectiveness, and robot performance stability.

### 4.4 Challenges and Technical Limitations

The project encountered computational constraints, training stability challenges, and limitations in simulating realistic physical interactions, highlighting the need for iterative tuning and comprehensive validation.

## Chapter 5: Conclusions and Recommendations

### 5.1 Summary of Achievements

The project successfully demonstrated the application of PPO-based DRL to autonomous sumo robot simulations, showing significant enhancements in competitive strategies and adaptability across different simulated scenarios.

### 5.2 Technical and Engineering Contributions

Key contributions include detailed design of simulation environments, optimized PPO training methodologies, insightful behavioral analyses, and rigorous experimental approaches enabling effective iterative improvements.

### 5.3 Limitations and Future Work

Recommendations include:

* **Reward Adjustments:** Increase rewards for successful knockouts, clearer player-opponent definitions, and optimized timestep penalties.
* **Environment Improvements:** Adjust agent friction or torque to enhance aggressive interactions, increase collision impact to promote proactive behaviors, and implement a structured self-play curriculum to improve training insights.

## References

Detailed academic citations and resources used throughout the project.

## Appendix

* URDF robot model code snippets
* Essential Python scripts from simulation and training environments
* Extensive logs and visual documentation supporting the analyses
