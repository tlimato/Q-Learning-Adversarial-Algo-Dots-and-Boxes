import numpy as np
from collections import defaultdict

GRID = 3
DOTS = GRID + 1
H = GRID * DOTS # horizontal line count
V = GRID * DOTS # vertical line count
N = H + V # total lines
BOXES = GRID * GRID

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
            row = ""
            for c in range(GRID):
                row += "+" + ("---" if self.lines[h_idx(r, c)] else "   ")
            print(row + "+")
            for c in range(DOTS):
                col = "|" if self.lines[v_idx(r, c)] else " "
                box = self.boxes[r * GRID + c] if c < GRID else 0
                col += " A " if box == 1 else " H " if box == -1 else "   "
                print(col, end="")
            print()
        print("+" + ("---+" * GRID))
    
    def done(self): return len(self.legal()) == 0
    def score(self): return (self.boxes == 1).sum(), (self.boxes == -1).sum()


class QLearner:
    """
    Actual Q learning class
    """
    def __init__(self, alpha=0.3, gamma=0.9, eps=1.0, eps_decay=0.9997, eps_min=0.05):
        self.Q = defaultdict(float)
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_decay, self.eps_min = eps, eps_decay, eps_min

    def _key(self, state_key, move): return state_key + move.to_bytes(2, 'little')

    def get_q(self, sk, m): return self.Q[self._key(sk, m)]

    def best(self, g):
        moves = g.legal()
        sk = g.key()
        return max(moves, key=lambda m: self.get_q(sk, m))

    def choose(self, g):
        return np.random.choice(g.legal()) if np.random.rand() < self.eps else self.best(g)

    def train(self, n=10000):
        # run n full games from start to finish
        for _ in range(n):
            g = DotsBoxes()
            traj = []  # stores every (state, move, reward, next_state) in this game

            # play out the game, recording each move
            while not g.done():
                sk, m = g.key(), self.choose(g)  # pick a move (random or greedy)
                ng, scored = g.step(m)           # apply it and get the new board

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

            # reduce randomness slightly so the agent explores less as it learns more and just "knows" the optimal move set
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def play_vs_human(self):
        g = DotsBoxes()
        while not g.done():
            g.render()
            if g.turn == 0:
                m = self.best(g)
                print(f"AI plays line {m}")
            else:
                m = int(input("Your move (line index): "))
            g, _ = g.step(m)
        a, b = g.score()
        print(f"Final — AI:{a}  Human:{b}")

if __name__ == "__main__":
    game = QLearner()
    game.play_vs_human()