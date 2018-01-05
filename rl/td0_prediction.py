from __future__ import print_function, division
from builtins import range


import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a, esp=0.1):
    if np.random.random() < (1 - esp):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_game(grid, policy):
    s = (2, 0)
    grid.set_state(s)
    states_and_rewards = [(s, 0)]
    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))
    return states_and_rewards


if __name__ == '__main__':
    grid = standard_grid()

    print("rewards:")
    print_values(grid.rewards, grid)

    policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
    }

    V = {s:0 for s in grid.all_states()}

    for t in range(1000):
        states_and_rewards = play_game(grid, policy)
        for idx in range(len(states_and_rewards)-1):
            s, _ = states_and_rewards[idx]
            s_prime, r = states_and_rewards[idx+1]
            V[s] = V[s] + ALPHA * ((r + GAMMA * V[s_prime]) - V[s])

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
