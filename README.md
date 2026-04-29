## Q-Learning in Dots and Boxes

The agent keeps a Q-table $Q : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ - a lookup from every (state, action) pair to a score representing how good that move is. Each state $s_k$ captures the current board lines and whose turn it is.

When choosing a move, the agent uses an **ε-greedy policy**: with probability $\varepsilon$ it explores by picking randomly, otherwise it exploits what it knows by taking the highest-scoring move:

$$\pi^*(s) = \arg\max_{m \in \mathcal{A}(s)} Q(s, m)$$

Each move earns a reward based on the box differential - boxes gained minus boxes given away:

$$r_t = \Delta a_{\text{boxes}} - \Delta b_{\text{boxes}}$$

At the end of the game a bonus of $\pm 2$ is added for a win or loss.

After each episode, the agent walks backwards through the moves it made and computes a target value for each one. For the last move, the target folds in the terminal bonus. For every earlier move, it looks ahead one step:

$$\text{target}_i = r_i + \gamma \max_{m'} Q(s_{i+1}, m')$$

where $\gamma = 0.9$ controls how much future rewards are discounted. Each Q-value is then nudged toward its target using the Bellman update:

$$Q(s_i, a_i) \leftarrow Q(s_i, a_i) + \alpha\,\delta_i$$

where $\alpha = 0.3$ is the learning rate and $\delta_i = \text{target}_i - Q(s_i, a_i)$ is the error between what the agent expected and what it should have expected.

Finally, $\varepsilon$ shrinks a little after each episode:

$$\varepsilon \leftarrow \max\!\left(\varepsilon_{\min},\; \varepsilon \cdot \lambda\right)$$

with $\lambda = 0.9997$ and $\varepsilon_{\min} = 0.05$, so the agent gradually stops exploring and starts trusting what it has learned.


References:
Debugging was assisted using Microsoft Copilot

## Citations - NOT ORGANIZED YET CLEAN UP
- [Epsilon-Greedy Algorithm in Reinforcement Learning](https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/) - explains the exploration vs exploitation tradeoff that drives the `choose()` 
- [Hugging Face Deep RL Course Unit 2](https://huggingface.co/learn/deep-rl-course/unit2/introduction) - covering TD learning and the Bellman update. `train()` loop trys to implement this
- [StatQuest YouTube Channel](https://www.youtube.com/@statquest) - Josh Starmer breakdowns of ML concepts including Q-learning
- [Sentdex RL Playlist](https://www.youtube.com/@sentdex) - hands-on Python RL series that builds a tabular Q-learning agent
- [freeCodeCamp NumPy Tutorial](https://www.youtube.com/watch?v=QUT1VHiLmmI) - 1 hr beginner video covering the core NumPy operations. referencced this plus the actual docs.
- [W3Schools NumPy](https://www.w3schools.com/python/numpy/default.asp) - quick reference for the `np.zeros`, `np.where`, and array methods used throughout my environment
- [freeCodeCamp RL Handbook](https://www.freecodecamp.org/news/the-complete-guide-to-reinforcement-learning/) - actually simple guide covering reward shaping, the Bellman equation, and discount factors
- [NeuralNine YouTube](https://www.youtube.com/@NeuralNine) - his game AI series trains an agent on a custom environment using the same `environment class + learner class` pattern I used although the context is different
- [Real Python RL Series](https://realpython.com/python-reinforcement-learning/) - walks through building a custom game environment and tabular Q-learning agent from scratch in Python
- [Towards Data Science on Medium](https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c) - written walkthrough of tabular Q-learning with annotated code examples