import time

import numpy as np
import funcy as fy
from cachetools import cached


def simulation_step(d_t, iterations, pos, prev_pos, edges, constraints, weights, coloring):
    d_t = d_t / iterations
    max_residual = 0
    for _ in range(iterations):
        pos = pos + (pos - prev_pos + d_t ** 2 * np.array([0, 0, -9.81])) * weights[:, np.newaxis]
        pos, max_residual = solve_edge_constraints(pos, edges, weights, coloring, constraints)
        prev_pos = pos

    print(f'Max residual: {max_residual}')

    return pos


def solve_edge_constraints(pos, edges, weights, coloring, constraints):
    max_residual = 0
    for color in range(np.max(coloring) + 1):
        d, e1, e2, w1, w2, w_sum = static_variables(color, coloring, constraints, edges, weights)

        diff = pos[e1] - pos[e2]

        dist = np.linalg.norm(diff, axis=1)

        s = (dist - d) / w_sum
        corr = diff / dist[:, np.newaxis] * s[:, np.newaxis]

        max_residual = max(max_residual, np.linalg.norm(corr))

        pos[e1] -= w1[:, np.newaxis] * corr
        pos[e2] += w2[:, np.newaxis] * corr

    return pos, max_residual


def static_variables(color, coloring, constraints, edges, weights):
    color_edges = edges[coloring == color]
    e1 = color_edges[:, 0]
    e2 = color_edges[:, 1]
    d = constraints[coloring == color]

    w1 = weights[e1]
    w2 = weights[e2]

    w_sum = w1 + w2
    w_sum[w_sum == 0] = 1

    return d, e1, e2, w1, w2, w_sum
