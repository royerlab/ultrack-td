#pragma once

#include <unordered_map>
#include <vector>
#include <set>

struct RAG {
    std::unordered_map<long, std::vector<long>> neighbors;
    std::unordered_map<std::pair<long, long>, float> weights;
    std::unordered_map<long, long> num_pixels;

    void add_edge(long i, long j, float weight)
    {
        if (i > j) std::swap(i, j);

        neighbors[i].push_back(j);
        neighbors[j].push_back(i);

        std::pair<long, long> key = {i, j};

        if (weights.find(key) == weights.end()) {
            weights[key] = weight;
        } else {
            weights[key] = std::min(weights[key], weight);
        }
    }

/*
    std::vector<std::set<long>> k_connected_sets(
        int k,
        int min_num_pixels,
        int max_num_pixels,
        float min_frontier
    )
    {
        std::vector<std::set<long>> sets;
    }
*/
};
