from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from td0_prediction import random_action

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def max_dict(d):
    idx = np.argmax([*d.values()])
    return list(d.items())[idx]

if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.1)
    print("rewards:")
    print_values(grid.rewards, grid)
    states = grid.all_states()
    Q = {s: {a: 0 for a in ALL_POSSIBLE_ACTIONS} for s in states}
    update_counts_sa = {s: {a: 1.0 for a in ALL_POSSIBLE_ACTIONS} for s in states}
    update_counts = {}

    t = 1.0
    deltas = []
    for it in range(10000):
        if it % 100 == 0:
            t += 1e-2
        if it % 2000 == 0:
            print("it:", it)

        s = (2, 0)
        grid.set_state(s)
        a, _ = max_dict(Q[s])
        biggest_change = 0
        while not grid.game_over():
            a = random_action(a, esp=0.5/t)
            r = grid.move(a)
            s2 = grid.current_state()
            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005
            old_qsa = Q[s][a]
            a2, max_q_s2a2 = max_dict(Q[s2])
            Q[s][a] = Q[s][a] + alpha * ((r + GAMMA * max_q_s2a2) - Q[s][a])
            biggest_change = max(biggest_change, Q[s][a] - old_qsa)

            s, a = s2, a2
        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

    print("update counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = float(v) / total
        print_values(update_counts, grid)

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
