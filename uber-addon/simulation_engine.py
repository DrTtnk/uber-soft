import numpy as np


def simulation_step(d_t, iterations, pos, prev_pos, edges, constraints, weights, colors):
    for _ in range(iterations):
        for edge, constraint in zip(edges, constraints):
            solve_edge_constraint(pos, edge, weights, constraint)

    new_positions = pos + (pos - prev_pos + d_t ** 2 * np.array([0, 0, -9.81])) * weights[:, np.newaxis]

    return new_positions


def solve_edge_constraints(pos, edges, weights, colors, d):
    num_colors = np.max(colors) + 1
    for color in range(num_colors):
        color_edges = np.where(colors == color)[0]
        p1 = pos[color_edges[0]]
        p2 = pos[color_edges[1]]
        w1 = weights[color_edges[0]]
        w2 = weights[color_edges[1]]

        corr = (w1 + w2) * (np.linalg.norm(p1 - p2) - d) * (p1 - p2) / np.linalg.norm(p1 - p2)

        pos[color_edges[0]] -= w1 / corr
        pos[color_edges[1]] += w2 / corr


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
