#pragma once

#include <vector>
#include <numeric>


template <typename WeightType = float>
struct BinaryTree {

    private:

    int current_node;
    std::vector<int> parent_;
    std::vector<int> children_;
    std::vector<WeightType> weight_;

    public:

    int num_leaves;

    BinaryTree(int n) :
    num_leaves(n), current_node(n),
    parent_(2 * n - 1), children_(2 * (n - 1), -1), weight_(n - 1, WeightType{})
    {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    inline int left_child(int n) const noexcept
    {
        int i = n - num_leaves;
        return children_[2 * i];
    }

    inline int right_child(int n) const noexcept
    {
        int i = n - num_leaves;
        return children_[2 * i + 1];
    }

    inline int parent(int n) const noexcept
    {
        return parent_[n];
    }

    inline WeightType weight(int n) const noexcept
    {
        return weight_[n - num_leaves];
    }

    inline int add_node(int left_child, int right_child, WeightType weight) noexcept
    {
        int p = current_node;

        parent_[p] = p;
        parent_[left_child] = p;
        parent_[right_child] = p;

        int i = p - num_leaves;
        children_[2 * i] = left_child;
        children_[2 * i + 1] = right_child;

        weight_[i] = weight;

        current_node++;
        return p;
    }
};
