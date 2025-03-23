# Deep-Q-learning-With-Atari

# Deep Q-Network (DQN) Atari Agent

## Introduction
This project implements a Deep Q-Network (DQN) to train an agent to play an Atari game. The agent is trained using reinforcement learning and interacts with the environment to maximize its score.

## Project Structure
- `train.py`: Script for training the DQN agent.
- `play.py`: Script to test and visualize the trained agent.
- `README.md`: Documentation for the project.
- `video/`: Directory containing a demonstration of the agent playing in the Atari environment.

## Installation
To run this project, ensure you have the required dependencies installed:
```bash
pip install gym[atari] stable-baselines3 opencv-python
```

## Training the Agent
To train the agent, run:
```bash
python train.py
```
This script will initialize a DQN model and train it on the specified Atari environment.

## Playing with the Trained Agent
To test the trained agent, run:
```bash
python play.py
```
This will load the trained model and let the agent play in the environment while rendering the gameplay.

## Hyperparameter Tuning Results
The following table presents the results of different hyperparameter configurations tested during training:

| Hyperparameter Set | Learning Rate (lr) | Gamma | Batch Size | Epsilon Start | Epsilon End | Exploration Fraction | Train Frequency | Episode Scores (Ep1-Ep5) | Observations |
|--------------------|-------------------|-------|------------|---------------|-------------|---------------------|---------------|--------------------------|-------------|
| Baseline (Initial Parameters) | 1e-4 | 0.99  | 32  | 1.0 | 0.05 | 0.1 | 4 | 1, 3, 3, 3, 2 | Moderate reward, slight inconsistency in scores. |
| Experiment 1 (Optimized Params) | 5e-4 | 0.95 | 64  | 0.9 | 0.02 | 0.2 | 8 | 3, 3, 3, 3, 3 | Stable performance, consistently high scores. |
| Experiment 2 (Further Adjustments) | 5e-4 | 0.95 | 64  | 0.9 | 0.02 | 0.2 | 8 | 3, 2, 3, 3, 3 | Still good, but one episode dipped to 2, showing slight instability. |
| Experiment 3 (Higher Learning Rate) | 7e-4 | 0.95 | 64  | 0.9 | 0.02 | 0.2 | 8 | 3, 1, 3, 3, 3 | Mostly good, but the drop to 1 in Episode 2 suggests instability. |

## Observations & Insights
1. **Baseline Model:** Showed moderate reward but inconsistency in performance.
2. **Experiment 1:** Improved stability and consistently higher scores.
3. **Experiment 2:** Performed well, but had one episode with a slight drop in performance.
4. **Experiment 3:** Increased learning rate led to some instability, reducing score consistency.

### **Conclusion**
From the experiments, **Experiment 1** provided the best balance between learning stability and performance. Increasing the learning rate too much (Experiment 3) led to instability, while the baseline was inconsistent.

## Video Demonstration
A video showing the agent playing in the Atari environment can be found in the `video/` directory or accessed [here](#).

## Future Improvements
- Experiment with different exploration strategies, such as epsilon decay schedules.
- Implement Prioritized Experience Replay for better sample efficiency.
- Fine-tune train frequency and batch size to optimize learning stability.

---

**Author:** Samuel Babalola
**Contact:** s.babalola@alustudent.com

