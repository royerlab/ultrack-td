#pragma once

#include <vector>
#include <unordered_map>

struct BiMap {
    private:

    std::vector<int> forward;
    std::unordered_map<int, int> backward;

    public:
    BiMap(const std::vector<int> &values) : forward(values.size()), backward(values.size())
    {
        for (int i = 0; i < values.size(); i++)
        {
            forward[i] = values[i];
            backward[values[i]] = i;
        }
    }

    template<typename Container>
    std::vector<int> apply_forward(const Container &values) {
        std::vector<int> result;
        result.reserve(values.size());
        for (const auto& val : values) {
            result.push_back(forward[val]);
        }
        return result;
    }

    std::vector<int> apply_backward(const std::vector<int> &values) {
        std::vector<int> result;
        result.reserve(values.size());
        for (int i = 0; i < values.size(); i++) {
            result.push_back(backward[values[i]]);
        }
        return result;
    }
};
