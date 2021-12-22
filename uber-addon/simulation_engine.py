import numpy as np


def simulation_step(d_t, iterations, pos, prev_pos, edges, constraints, weights, coloring):
    d_t = d_t / iterations
    max_residual = 0
    for _ in range(iterations):
        pos = pos + (pos - prev_pos + d_t ** 2 * np.array([0, 0, -9.81])) * weights[:, np.newaxis]
        pos, max_residual = solve_edge_constraints(pos, edges, weights, coloring, constraints)
        prev_pos = pos

    print(f'Max residual: {max_residual}')

    return pos


def solve_edge_constraints(pos, edges, weights, coloring, rest_length):
    max_residual = 0
    for color in range(np.max(coloring) + 1):
        d, e1, e2, w1, w2, w_sum = edge_static_variables(color, coloring, rest_length, edges, weights)

        diff = pos[e1] - pos[e2]

        dist = np.linalg.norm(diff, axis=1)

        s = (dist - d) / w_sum
        corr = diff / dist[:, np.newaxis] * s[:, np.newaxis]

        max_residual = max(max_residual, np.linalg.norm(corr))

        pos[e1] -= w1[:, np.newaxis] * corr
        pos[e2] += w2[:, np.newaxis] * corr

    return pos, max_residual


def edge_static_variables(color, coloring, rest_length, edges, weights):
    color_edges = edges[coloring == color]
    e1 = color_edges[:, 0]
    e2 = color_edges[:, 1]
    d = rest_length[coloring == color]

    w1 = weights[e1]
    w2 = weights[e2]

    return d, e1, e2, w1, w2, w1 + w2


def solver_tetrahedron(pos, tets, weights, coloring, rest_volume):
    # https://github.com/Q-Minh/position-based-dynamics/blob/master/src/xpbd/edge_length_constraint.cpp
    # https://matthias-research.github.io/pages/publications/strainBasedDynamics.pdf
    for color in range(np.max(coloring) + 1):
        color_tets = tets[coloring == color]
        w = weights[color_tets]

        p = pos[color_tets]

        vol = np.abs((1. / 6.) * np.sum(np.cross(p[1] - p[0], p[2] - p[0], axis=1) * p[3] - p[0], axis=1))

        grad0 = (1. / 6.) * np.cross(p[1] - p[2], p[3] - p[2], axis=1)
        grad1 = (1. / 6.) * np.cross(p[2] - p[0], p[3] - p[0], axis=1)
        grad2 = (1. / 6.) * np.cross(p[0] - p[1], p[3] - p[1], axis=1)
        grad3 = (1. / 6.) * np.cross(p[1] - p[0], p[2] - p[0], axis=1)

        weighted_sum_of_gradients = w[0] * np.sum(grad0 ** 2, axis=1) + \
                                    w[1] * np.sum(grad1 ** 2, axis=1) + \
                                    w[2] * np.sum(grad2 ** 2, axis=1) + \
                                    w[3] * np.sum(grad3 ** 2, axis=1)

        s = -(vol - rest_volume) / weighted_sum_of_gradients

        pos[color_tets] += w * np.concatenate([grad0, grad1, grad2, grad3], axis=1) * s[:, np.newaxis]

    return pos


def continuous_collision_detection(p0, p1,
                                   a0, b0, c0,
                                   a1, b1, c1):
    ap0 = p0 - a0
    ab0 = b0 - a0
    ac0 = c0 - a0

    ap1 = p1 - a1
    ab1 = b1 - a1
    ac1 = c1 - a1

    dap = ap1 - ap0
    dab = ab1 - ab0
    dac = ac1 - ac0

    n_0 = np.cross(ab0, ac0)
    n_1 = np.cross(dab, ac0) + np.cross(ab0, dac)
    n_2 = np.cross(dab, dac)

    a = np.dot(dap, n_2)
    b = np.dot(ap0, n_2) + np.dot(dap, n_1)
    c = np.dot(ap0, n_1) + np.dot(dap, n_0)
    d = np.dot(ap0, n_0)

    roots = np.roots([a, d, c, b])
    roots = roots[np.isreal(roots) & (roots > 0) & (roots < 1)]

    def validate(t):
        return validate_t(ap0, ab0, ac0, dap, dab, dac, t)

    return next(validate(r) for r in roots)


def validate_t(ap0, ab0, ac0, dAP, dAB, dAC, t):
    if t <= 0 or t >= 1:
        return False

    APt = ap0 + dAP * t
    ABt = ab0 + dAB * t
    ACt = ac0 + dAC * t

    u = np.cross(ABt, ACt)
    v = np.cross(ACt, APt)
    w = np.cross(APt, ABt)

    if np.dot(u, v) < 1e-6:
        return False

    if np.dot(u, w) < 1e-6:
        return False

    return True

