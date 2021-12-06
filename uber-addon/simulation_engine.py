import numpy as np


def simulation_step(d_t, iterations, pos, prev_pos, edges, constraints, weights, coloring):
    pos = pos + (pos - prev_pos + d_t ** 2 * np.array([0, 0, -9.81])) * weights[:, np.newaxis]

    for _ in range(iterations):
        pos = solve_edge_constraints(pos, edges, weights, coloring, constraints)

    return pos


def solve_edge_constraints(pos, edges, weights, coloring, constraints):
    max_corr = 0
    for color in range(np.max(coloring) + 1):
        color_edges = edges[coloring == color]

        if len(color_edges) == 0:
            continue

        e1 = color_edges[:, 0]
        e2 = color_edges[:, 1]
        p1 = pos[e1]
        p2 = pos[e2]
        w1 = weights[e1]
        w2 = weights[e2]
        d = constraints[coloring == color]

        dist = np.linalg.norm(p1 - p2, axis=1)

        n = (p1 - p2) / dist[:, np.newaxis]
        s = (dist - d) / (w1 + w2)
        corr = n * s[:, np.newaxis]

        max_corr = max(max_corr, np.max(np.abs(corr)))

        pos[e1] -= w1[:, np.newaxis] * corr
        pos[e2] += w2[:, np.newaxis] * corr

    print(f'Max correction: {max_corr}')

    return pos


def solve_edge_constraint(pos, edge, weights, d):
    p1 = pos[edge[0]]
    p2 = pos[edge[1]]
    w1 = weights[edge[0]]
    w2 = weights[edge[1]]

    if w1 == 0 and w2 == 0:
        return

    corr = (w1 + w2) * (np.linalg.norm(p1 - p2) - d) * (p1 - p2) / np.linalg.norm(p1 - p2)

    pos[edge[0]] -= w1 / corr
    pos[edge[1]] += w2 / corr
