#pragma once

#include <unordered_map>
#include <vector>
#include <list>
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

    float get_weight(long i, long j)
    {
        if (i > j) std::swap(i, j);
        return weights[{i, j}];
    }

    std::list<std::set<long>> _expand(
        int root,
        const std::set<long> &cur_set,
        const std::vector<long> &candidates,
        int cur_num_pixels,
        int min_num_pixels,
        int max_num_pixels,
        float min_frontier
    ) {
        std::list<std::set<long>> sets;

        if (cur_num_pixels < max_num_pixels)
        {
            // adding to the list if the current set meets the size constraints
            if (cur_num_pixels >= min_num_pixels) {
                sets.push_back(cur_set);
            }

            for (int i = 0; i < candidates.size(); i++)
            {
                long c = candidates[i];
                long new_num_pixels = cur_num_pixels + num_pixels[c];
                // stop if the new set exceeds the size constraints
                if (new_num_pixels < max_num_pixels)
                    continue;

                // expand the current set with the new candidate
                std::set<long> new_set = cur_set;
                new_set.insert(c);

                std::vector<long> new_candidates(candidates.begin() + i + 1, candidates.end());
                for (long n : neighbors[c]) {
                    // add the new candidate if it meets the size constraints and is not already in the set
                    if (n > root &&
                        new_set.find(n) == new_set.end() &&
                        get_weight(c, n) >= min_frontier)
                    {
                        new_candidates.push_back(n);
                    }
                }

                sets.splice(
                    sets.end(),
                    _expand(
                        root, new_set, new_candidates, new_num_pixels,
                        min_num_pixels, max_num_pixels, min_frontier
                    )
                );
            }
        }

       return sets;
    }

    std::list<std::set<long>> connected_subgraphs(
        int min_num_pixels,
        int max_num_pixels,
        float min_frontier
    )
    {
        std::list<std::set<long>> sets;
        for (const std::pair<long, std::vector<long>> &entry : neighbors)
        {
            long cur_idx = entry.first;
            const std::set<long> &cur_set = {cur_idx};

            std::vector<long> candidates;
            for (long n : entry.second) {
                if (n > cur_idx && weights[{cur_idx, n}] >= min_frontier)
                    candidates.push_back(n);
            }

            const int cur_num_pixels = num_pixels[cur_idx];
            sets.splice(
                sets.end(),
                _expand(
                    cur_idx, cur_set, candidates, cur_num_pixels,
                    min_num_pixels, max_num_pixels, min_frontier
                )
            );
        }
        return sets;
    }
};
