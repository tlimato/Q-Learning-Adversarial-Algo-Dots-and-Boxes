# Author: Tyson Limato
# Date: Apr 25th, 2026
"""
Purpose:
    Create an as close to scratch as is reasonable implimentation of adversarial Q-Learning Agents for the children's game Dots and Boxes.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
from collections import defaultdict


GRID = 3
DOTS = GRID + 1
H = GRID * DOTS
V = GRID * DOTS
N = H + V
BOXES = GRID * GRID
TIME_ZONE = datetime.timezone.utc
TIME_STAMP = datetime.datetime.now(tz=TIME_ZONE)
GAME_COUNT = 100000

#----game colors----
RESET  = "\033[0m"
BLUE   = "\033[94m"   # AI boxes
RED    = "\033[91m"   # Human boxes
YELLOW = "\033[93m"   # drawn lines
GRAY   = "\033[90m"   # empty lines/dots

def h_idx(r, c): return r * GRID + c
def v_idx(r, c): return H + r * DOTS + c
def box_edges(r, c): return [h_idx(r,c), h_idx(r+1,c), v_idx(r,c), v_idx(r,c+1)]


class DotsBoxes:
    def __init__(self):
        self.lines = np.zeros(N, dtype=np.int8)
        self.boxes = np.zeros(BOXES, dtype=np.int8)
        self.turn = 0

    def clone(self):
        g = DotsBoxes()
        g.lines = self.lines.copy()
        g.boxes = self.boxes.copy()
        g.turn = self.turn
        return g

    def key(self):
        return self.lines.tobytes() + bytes([self.turn])

    def legal(self):
        return np.where(self.lines == 0)[0].tolist()

    def step(self, line):
        g = self.clone()
        g.lines[line] = 1
        scored = 0
        for r in range(GRID):
            for c in range(GRID):
                idx = r * GRID + c
                if g.boxes[idx] == 0 and all(g.lines[e] for e in box_edges(r, c)):
                    g.boxes[idx] = 1 if g.turn == 0 else -1
                    scored += 1
        if scored == 0:
            g.turn = 1 - g.turn
        return g, scored

    def render(self):
        for r in range(GRID):
            # top edge of each cell in this row
            row = ""
            for c in range(GRID):
                if self.lines[h_idx(r, c)]:
                    row += YELLOW + "+---" + RESET
                else:
                    row += GRAY + "+   " + RESET
            print(row + GRAY + "+" + RESET)

            # vertical edges and box contents
            for c in range(DOTS):
                if self.lines[v_idx(r, c)]:
                    print(YELLOW + "|" + RESET, end="")
                else:
                    print(GRAY + " " + RESET, end="")
                if c < GRID:
                    box = self.boxes[r * GRID + c]
                    if box == 1:
                        print(BLUE + " A " + RESET, end="")
                    elif box == -1:
                        print(RED + " B " + RESET, end="")
                    else:
                        print("   ", end="")
            print()

        # bottom edge of the board
        bottom = ""
        for c in range(GRID):
            if self.lines[h_idx(GRID, c)]:
                bottom += YELLOW + "+---" + RESET
            else:
                bottom += GRAY + "+   " + RESET
        print(bottom + GRAY + "+" + RESET)

    def done(self): return len(self.legal()) == 0
    def score(self): return (self.boxes == 1).sum(), (self.boxes == -1).sum()


class QLearner:
    """
    Tabular Q-learning agent for Dots and Boxes.
    Uses epsilon-greedy exploration with multiplicative decay,
    and trains via backwards TD updates over full episode trajectories.
    """
    def __init__(self, alpha=0.3, gamma=0.9, eps=1.0, eps_decay=0.9997, eps_min=0.05):
        self.Q = defaultdict(float)
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_decay, self.eps_min = eps, eps_decay, eps_min

    def _key(self, state_key, move): return state_key + int(move).to_bytes(2, 'little')

    def get_q(self, sk, m): return self.Q[self._key(sk, m)]

    def best(self, g):
        moves = g.legal()
        sk = g.key()
        return max(moves, key=lambda m: self.get_q(sk, m))

    def choose(self, g):
        return np.random.choice(g.legal()) if np.random.rand() < self.eps else self.best(g)

    def save(self, path):
        # save Q-table and current epsilon to disk
        with open(path, "wb") as f:
            pickle.dump({"Q": dict(self.Q), "eps": self.eps}, f)
        print(f"Saved weights to {path}")

    def load(self, path):
        # load Q-table and epsilon from a previous training run
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(float, data["Q"])
        self.eps = data["eps"]
        print(f"Loaded weights from {path} — Q-states: {len(self.Q)}, ε: {self.eps:.4f}")

    def train(self, n=10000):
        # run n full games from start to finish
        for _ in range(n):
            g = DotsBoxes()
            traj = []  # stores every (state, move, reward, next_state) in this game

            # play out the game, recording each move
            while not g.done():
                sk, m = g.key(), self.choose(g)  # pick a move (random or greedy)
                ng, scored = g.step(m)            # apply it and get the new board

                # reward = boxes we just gained minus boxes opponent just gained
                sa, sb = g.score(); na, nb = ng.score()
                r = (na - sa) - (nb - sb)

                traj.append((sk, m, r, ng))  # save this step for later learning
                g = ng                        # advance to the new board state

            # assign a win/loss/draw bonus based on the final score
            fa, fb = g.score()
            terminal = 2 if fa > fb else -2 if fb > fa else 0

            # walk backwards through the game and update Q-values
            # backwards pass lets each step learn from what came after it
            for i in range(len(traj) - 1, -1, -1):
                sk, m, r, ng = traj[i]
                nmoves = ng.legal()

                # best possible Q-value we could get from the next state
                next_max = max((self.get_q(ng.key(), nm) for nm in nmoves), default=0)

                # for the last move, fold in the terminal bonus; otherwise discount future value
                target = r + (terminal if i == len(traj)-1 else self.gamma * next_max)

                # nudge Q toward the target by a fraction (alpha) of the error
                self.Q[self._key(sk, m)] += self.alpha * (target - self.get_q(sk, m))

            # reduce randomness slightly so the agent explores less as it learns more
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def play_vs_human(self):
        g = DotsBoxes()
        while not g.done():
            g.render()
            if g.turn == 0:
                m = self.best(g)
                print(f"AI plays line {m}")
            else:
                print("Enter move as: h <row> <col>  or  v <row> <col>")
                print("  h = horizontal line,  v = vertical line")
                parts = input("Your move: ").strip().split()
                direction, r, c = parts[0], int(parts[1]), int(parts[2])
                m = h_idx(r, c) if direction == "h" else v_idx(r, c)
            g, _ = g.step(m)
        a, b = g.score()
        print(f"Final — AI:{a}  Human:{b}")


def train_two_agents(agent_a, agent_b, n=10000, log_every=500):
    """
    Run n games where agent_a plays as player 0 and agent_b as player 1.
    Collects stats every log_every episodes for plotting.
    """
    wins_a = wins_b = draws = 0
    history = []

    for ep in range(n):
        g = DotsBoxes()
        traj_a = []
        traj_b = []
        # randomly assign which agent goes first each episode
        first_player = np.random.randint(0, 2)  # 0 = agent_a goes first, 1 = agent_b goes first
        while not g.done():
            sk = g.key()
            # Initially I was testing with agent_a going first, however after
            # enough games gradually agent b learned by going second how to win 75% of the time
            # m = agent_a.choose(g) if g.turn == 0 else agent_b.choose(g) # original code line
            # swap agent assignments based on who was randomly chosen to go first
            if first_player == 0:
                m = agent_a.choose(g) if g.turn == 0 else agent_b.choose(g)
            else:
                m = agent_b.choose(g) if g.turn == 0 else agent_a.choose(g)

            ng, _ = g.step(m)

            sa, sb = g.score()
            na, nb = ng.score()
            r_a = (na - sa) - (nb - sb)
            r_b = (nb - sb) - (na - sa)

            if g.turn == 0:
                traj_a.append((sk, m, r_a, ng))
            else:
                traj_b.append((sk, m, r_b, ng))

            g = ng

        fa, fb = g.score()
        if fa > fb:
            wins_a += 1
            term_a, term_b = 2, -2
        elif fb > fa:
            wins_b += 1
            term_a, term_b = -2, 2
        else:
            draws += 1
            term_a = term_b = 0

        for i in range(len(traj_a) - 1, -1, -1):
            sk, m, r, ng = traj_a[i]
            nmoves = ng.legal()
            next_max = max((agent_a.get_q(ng.key(), nm) for nm in nmoves), default=0)
            target = r + (term_a if i == len(traj_a)-1 else agent_a.gamma * next_max)
            agent_a.Q[agent_a._key(sk, m)] += agent_a.alpha * (target - agent_a.get_q(sk, m))

        for i in range(len(traj_b) - 1, -1, -1):
            sk, m, r, ng = traj_b[i]
            nmoves = ng.legal()
            next_max = max((agent_b.get_q(ng.key(), nm) for nm in nmoves), default=0)
            target = r + (term_b if i == len(traj_b)-1 else agent_b.gamma * next_max)
            agent_b.Q[agent_b._key(sk, m)] += agent_b.alpha * (target - agent_b.get_q(sk, m))

        agent_a.eps = max(agent_a.eps_min, agent_a.eps * agent_a.eps_decay)
        agent_b.eps = max(agent_b.eps_min, agent_b.eps * agent_b.eps_decay)

        # snapshot stats every log_every episodes
        if (ep + 1) % log_every == 0:
            total = wins_a + wins_b + draws
            history.append({
                "ep":         ep + 1,
                "win_rate_a": wins_a / total,
                "win_rate_b": wins_b / total,
                "draw_rate":  draws  / total,
                "epsilon":    agent_a.eps,
                "qtable":     len(agent_a.Q) + len(agent_b.Q),
            })
            print(f"Episode {ep+1}/{n} — A: {wins_a}  B: {wins_b}  Draws: {draws}  ε: {agent_a.eps:.4f}  Q-states: {history[-1]['qtable']}")

    return wins_a, wins_b, draws, history

def plot_training(history):
    """
    Plot win rates, epsilon decay, and Q-table growth
    from the history list collected during training.
    """
    episodes     = [h["ep"]        for h in history]
    win_rate_a   = [h["win_rate_a"] for h in history]
    win_rate_b   = [h["win_rate_b"] for h in history]
    draw_rate    = [h["draw_rate"]  for h in history]
    epsilons     = [h["epsilon"]    for h in history]
    qtable_sizes = [h["qtable"]     for h in history]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Q-Learning Training Progress — Dots and Boxes", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- win rate over time ---
    ax1 = fig.add_subplot(gs[0, :])  # full top row
    ax1.plot(episodes, win_rate_a, label="Agent A win rate", color="#4C8BF5", linewidth=2)
    ax1.plot(episodes, win_rate_b, label="Agent B win rate", color="#E05555", linewidth=2)
    ax1.plot(episodes, draw_rate,  label="Draw rate",        color="#A0A0A0", linewidth=1.5, linestyle="--")
    ax1.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")  # 50% reference line
    ax1.set_title("Win rate over time (cumulative)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- epsilon decay ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(episodes, epsilons, color="#F5A623", linewidth=2)
    ax2.fill_between(episodes, epsilons, alpha=0.15, color="#F5A623")
    ax2.set_title("Epsilon decay (exploration rate)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("ε")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # --- Q-table size growth ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(episodes, qtable_sizes, color="#7ED321", linewidth=2)
    ax3.fill_between(episodes, qtable_sizes, alpha=0.15, color="#7ED321")
    ax3.set_title("Q-table growth (unique state-action pairs)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Q-table entries")
    ax3.grid(True, alpha=0.3)

    plt.savefig(f"training_progress_{TIME_STAMP}.png", dpi=300, bbox_inches="tight")
    print(f"Saved plot to training_progress_{TIME_STAMP}_{int(GAME_COUNT/1000)}k_games.png")
    plt.show()

if __name__ == "__main__":
    agent_a = QLearner()
    agent_b = QLearner()

    print("Training two agents against each other...")
    wins_a, wins_b, draws, history = train_two_agents(agent_a, agent_b, n=GAME_COUNT, log_every=100)
    print(f"\nFinal — A wins: {wins_a}  B wins: {wins_b}  Draws: {draws}")

    agent_a.save(f"agent_a_{int(GAME_COUNT/1000)}k_games_{TIME_STAMP}.pkl")
    agent_b.save(f"agent_b_{int(GAME_COUNT/1000)}k_games_{TIME_STAMP}.pkl")

    plot_training(history)