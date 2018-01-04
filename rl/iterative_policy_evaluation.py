from __future__ import print_function, division
from builtins import range


import numpy as np
from grid_world import standard_grid, negative_grid

SMALL_ENOUGH = 1e-3
def print_values(V, g):
  for i in range(g.width):
    print("---------------------------")
    for j in range(g.height):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")

def print_policy(P, g):
    for i in range(g.width):
        print("-------------")
        for j in range(g.height):
            a = P.get((i,j), ' ')
            print("{} |".format(a), end="")
        print("")

if __name__ == '__main__':
    grid = negative_grid()
    states = grid.all_states()
    V = {}
    for s in states:
        V[s] = 0
    gamma = 1.0
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]
            if s in grid.actions:
                new_v = 0
                p_a = 1.0 / len(grid.actions[s])
                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - new_v))
        if biggest_change < SMALL_ENOUGH:
            break
    print("values for uniformly random actions:")
    print_values(V, grid)
    print("\n\n")
    print_policy(grid.actions, grid)
