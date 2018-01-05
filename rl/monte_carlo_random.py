from __future__ import print_function, division
from builtins import range


import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def random_action(a):
    return np.random.choice([a, np.random.choice([action for action in ALL_POSSIBLE_ACTIONS if action != a])],p=[0.5,0.5])

def play_game(grid, policy):
  start_states = [*grid.actions]
  start_index = np.random.choice(len(start_states))
  grid.set_state(start_states[start_index])

  s = grid.current_state()
  states_and_rewards = [(s, 0)]
  while not grid.game_over():
      a = random_action(policy[s])
      r = grid.move(a)
      states_and_rewards.append((s, r))
      s = grid.current_state()

  G = 0
  states_and_returns = []
  for s, r in reversed(states_and_rewards):
      G = r + GAMMA * G
      states_and_returns.append((s, G))
  states_and_returns.reverse()
  return states_and_returns


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
    (1, 2): 'U',
    (2, 1): 'L',
    (2, 2): 'U',
    (2, 3): 'L',
  }

  V = {}
  returns = {}
  states = grid.all_states()
  for s in states:
    if s in grid.actions:
      returns[s] = []
    else:
      V[s] = 0

  for t in range(100):
    states_and_returns = play_game(grid, policy)
    seen_states = set()
    for s, G in states_and_returns:
      if s not in seen_states:
        returns[s].append(G)
        V[s] = np.mean(returns[s])
        seen_states.add(s)
  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)
