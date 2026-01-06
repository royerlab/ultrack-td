#pragma once

#include "rag.h"
#include "segment.h"


template <typename S, typename T>
std::vector<Segment> combinatorial_hypotheses(
    const nb::ndarray<S>& superpixels,
    const nb::ndarray<T>& contours,
    int min_num_pixels,
    int max_num_pixels,
    float min_frontier
) {
    size_t depth = superpixels.shape(0);
    size_t height = superpixels.shape(1);
    size_t width = superpixels.shape(2);

    S *spp_data = superpixels.data();
    T *contours_data = contours.data();

    // the graph is undirected, so we can only add edges in one direction
    constexpr std::array<int, 9> offsets = {
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
    };

    RAG rag;

    for (int z = 0; z < depth; z++) {
        long z_step = z * height * width;
        for (int y = 0; y < height; y++) {
            long y_step = y * width;
            for (int x = 0; x < width; x++) {
                long idx = z_step + y_step + x;
                if (spp_data[idx])
                {
                    rag.num_pixels[idx]++;
                    for (int i = 0; i < 3; i++) {
                        long nz = z + offsets[i * 3];
                        long ny = y + offsets[i * 3 + 1];
                        long nx = x + offsets[i * 3 + 2];
                        if (nz >= 0 && nz < depth &&
                            ny >= 0 && ny < height &&
                            nx >= 0 && nx < width)
                        {
                            long nidx = nz * height * width + ny * width + nx;
                            if (spp_data[nidx] && spp_data[idx] != spp_data[nidx])
                            {
                                float weight = 0.5f * (contours_data[idx] + contours_data[nidx]);
                                rag.add_edge(idx, nidx, weight);
                            }
                        }
                    }
                }
            }
        }
    }

    std::list<std::set<long>> subgraphs = rag.connected_subgraphs(min_num_pixels, max_num_pixels, min_frontier);

    return segments;
}
