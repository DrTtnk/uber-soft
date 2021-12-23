#include <pybind11/pybind11.h>
#include "glm/glm//glm.hpp"

namespace py = pybind11;

struct EdgeConstraint {
    float d;
    int e1;
    int e2;
    float w1;
    float w2;
    float w_sum;
};

// Define struct for edge constraints
struct EdgeConstraints {
    std::vector<int>   e1;
    std::vector<int>   e2;
    std::vector<float> w1;
    std::vector<float> w2;
    std::vector<float> w_sum;
    std::vector<float> d;

    EdgeConstraints(const std::vector<int>   &edges,
                    const std::vector<float> &weights,
                    const std::vector<float> &rest_length,
                    const std::vector<int>   &coloring,
                    const int                color) {
        for (auto i{0}; i < edges.size(); i++) {
            if (coloring[i] == color) {
                auto e1_idx = edges[i];
                auto e2_idx = edges[i + 1];

                e1.emplace_back(e1_idx);
                e2.emplace_back(e2_idx);

                w1.emplace_back(weights[e1_idx]);
                w2.emplace_back(weights[e2_idx]);

                d.emplace_back(rest_length[i]);
            }
        }

        for (auto i{0}; i < e1.size(); i++) {
            w_sum.emplace_back(w1[i] == w2[i] ? 1.0 : w1[i] + w2[i]);
        }
    }

    auto at(const int i) { return EdgeConstraint{d[i], e1[i], e2[i], w1[i], w2[i], w_sum[i]}; }
};

auto solve_edge_constraint(std::vector<glm::vec3> &pos, const std::vector<EdgeConstraints> &constraints) {
    for (auto color_group : constraints) {
        for (auto i{0}; i < color_group.e1.size(); i++) {
            auto c = color_group.at(i);
            auto diff = pos[c.e1] - pos[c.e2];
            auto dist = glm::length(diff);
            auto s = (dist - c.d) / c.w_sum;
            auto corr = diff / dist * s;
            pos[c.e1] -= corr * c.w1;
            pos[c.e2] += corr * c.w2;
        }
    }

    return pos;
}

auto simulation_step(const float d_t,
                     const int   iterations,
                     std::vector<glm::vec3>        &pos,
                     std::vector<glm::vec3>        &prev_pos,
                     const std::vector<int>   &edges,
                     const std::vector<float> &constraints,
                     const std::vector<float> &weights,
                     const int                color_amount,
                     const std::vector<int>   &coloring) {

    auto edge_constraints = std::vector<EdgeConstraints>{};
    for (auto i{0}; i < color_amount; i++) {
        edge_constraints.emplace_back(EdgeConstraints(edges, weights, constraints, coloring, i));
    }

    auto sub_d_t = d_t / iterations;
    auto g = glm::vec3{0, 0, -9.8};
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < pos.size(); j++)
            pos[j] = pos[j] + (pos[j] - prev_pos[j] + g * sub_d_t * sub_d_t) * weights[j];
        prev_pos = pos = solve_edge_constraint(pos, edge_constraints);
    }
    return pos;
}

//PYBIND11_MODULE(cpp_to_python, handle){
//    handle.doc() = "pybind11 example plugin";
//    handle.def("add", &add, "A function which adds a scalar to an array");
//}