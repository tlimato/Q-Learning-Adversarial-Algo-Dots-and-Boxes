# Dots and Boxes — Adversarial Q-Learning

Two tabular Q-learning agents trained purely through self-play on a 3×3 Dots and Boxes board. The rules of the game, a Bellman update, and 100,000 episodes.

---

## How It Works

The agent keeps a Q-table $Q : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ and lookup from every (state, action) pair to a score representing how good that move is. Each state $s_k$ captures the current board lines and whose turn it is.

### Action Selection

When choosing a move, the agent uses an **ε-greedy policy**: with probability $\varepsilon$ it explores by picking randomly, otherwise it exploits what it knows by taking the highest-scoring move:

$$\pi^*(s) = \arg\max_{m \in \mathcal{A}(s)} Q(s, m)$$

### Reward

Each move earns a reward based on the box differential through boxes gained minus boxes given away:

$$r_t = \Delta a_{\text{boxes}} - \Delta b_{\text{boxes}}$$

At the end of the game a terminal bonus of $\pm 2$ is added for a win or loss (0 for a draw).

### Backward TD Update

After each episode, the agent walks backwards through the moves it made and computes a target value for each one. For the last move, the target folds in the terminal bonus. For every earlier move, it looks ahead one step:

$$\text{target}_i = r_i + \gamma \max_{m'} Q(s_{i+1}, m')$$

where $\gamma = 0.9$ controls how much future rewards are discounted. Each Q-value is then nudged toward its target using the Bellman update:

$$Q(s_i, a_i) \leftarrow Q(s_i, a_i) + \alpha\,\delta_i$$

where $\alpha = 0.3$ is the learning rate and $\delta_i = \text{target}_i - Q(s_i, a_i)$ is the TD error.

### Exploration Decay

$\varepsilon$ shrinks after each episode so the agent gradually stops exploring and starts trusting what it has learned:

$$\varepsilon \leftarrow \max\!\left(\varepsilon_{\min},\; \varepsilon \cdot \lambda\right)$$

with $\lambda = 0.9997$ and $\varepsilon_{\min} = 0.05$.

---

## Training Results

Three snapshots from training show the progression from pure random play to convergent strategy:

| Episodes | Behavior |
|----------|----------|
| 1,000 | ε ≈ 0.75, win rates indistinguishable from random, Q-table still growing linearly |
| 10,000 | ε at floor, Q-table plateauing near 900k entries, win rates just beginning to diverge |
| 100,000 | Agents converge to stable strategies; emergent behaviors (chain avoidance, sacrifice moves) visible in play |

> **Note on first-mover bias:** With a fixed turn order, Agent B consistently reached ~75% win rate. This is not because it was strategically superior, but because it learned to exploit Agent A's positional habits. Randomizing which agent moves first each episode eliminated this bias and produced genuinely symmetric learning.

---

## Usage

**Train two agents against each other:**
```python
agent_a = QLearner()
agent_b = QLearner()
wins_a, wins_b, draws, history = train_two_agents(agent_a, agent_b, n=100000)
```

**Save and load trained weights:**
```python
agent_a.save("agent_a_100k.pkl")
agent_a.load("agent_a_100k.pkl")
```

**Play against a trained agent:**
```python
agent_a.play_vs_human()
# Enter moves as: h <row> <col>  or  v <row> <col>
```

**Plot training progress:**
```python
plot_training(history)
```

---

## Hyperparameters

| Parameter | Value | Role |
|-----------|-------|------|
| `alpha` | 0.3 | Learning rate |
| `gamma` | 0.9 | Discount factor |
| `eps` | 1.0 | Initial exploration rate |
| `eps_decay` | 0.9997 | Multiplicative decay per episode |
| `eps_min` | 0.05 | Exploration floor |
| `GRID` | 3 | Board size (3×3 boxes, 4×4 dots) |
| `GAME_COUNT` | 100,000 | Training episodes |

---

## References

### Academic

- Bellman, R. (1957). *Dynamic programming*. Princeton University Press.
- Berlekamp, E. R. (2000). *The dots-and-boxes game: Sophisticated child's play*. A K Peters.
- Hu, J., & Wellman, M. P. (2003). Nash Q-learning for general-sum stochastic games. *Journal of Machine Learning Research*, *4*, 1039–1069.
- Littman, M. L. (1994). Markov games as a framework for multi-agent reinforcement learning. In *Proceedings of the 11th International Conference on Machine Learning* (pp. 157–163).
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, *518*(7540), 529–533. https://doi.org/10.1038/nature14236
- Silver, D., Huang, A., Maddison, C. J., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, *529*(7587), 484–489. https://doi.org/10.1038/nature16961
- Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). Mastering the game of Go without human knowledge. *Nature*, *550*(7676), 354–359. https://doi.org/10.1038/nature24270
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, *8*(3–4), 279–292. https://doi.org/10.1007/BF00992698
- Wilson, D. B. (2006). *Dots-and-boxes is PSPACE-complete* [Technical report]. Microsoft Research.

### Learning Resources

| Resource | What it covers |
|----------|---------------|
| [Epsilon-Greedy Algorithm — GeeksforGeeks](https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/) | Exploration vs. exploitation tradeoff behind `choose()` |
| [Hugging Face Deep RL Course, Unit 2](https://huggingface.co/learn/deep-rl-course/unit2/introduction) | TD learning and the Bellman update underlying the `train()` loop |
| [freeCodeCamp — Complete RL Handbook](https://www.freecodecamp.org/news/the-complete-guide-to-reinforcement-learning/) | Reward shaping, Bellman equation, and discount factors |
| [Real Python — RL Series](https://realpython.com/python-reinforcement-learning/) | Building a custom game environment and tabular Q-learner from scratch |
| [Towards Data Science — Beginner's Guide to Q-Learning](https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c) | Annotated walkthrough of tabular Q-learning |
| [StatQuest with Josh Starmer](https://www.youtube.com/@statquest) | Conceptual breakdowns of ML ideas including Q-learning |
| [Sentdex — RL Playlist](https://www.youtube.com/@sentdex) | Hands-on Python RL series building a tabular agent |
| [NeuralNine — Game AI Series](https://www.youtube.com/@NeuralNine) | `environment class + learner class` pattern (different game, same architecture) |
| [freeCodeCamp — NumPy Tutorial](https://www.youtube.com/watch?v=QUT1VHiLmmI) | Core NumPy operations used in the environment |
| [W3Schools — NumPy Reference](https://www.w3schools.com/python/numpy/default.asp) | Quick reference for `np.zeros`, `np.where`, and array methods |

### Tools

- Debugging assisted by [Microsoft Copilot](https://copilot.microsoft.com)
