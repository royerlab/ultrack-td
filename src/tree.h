#include <vector>
#include <numeric>


struct BinaryTree {

    private:

    int num_leaves;
    int current_node;
    std::vector<int> parent_;
    std::vector<int> children_;
    std::vector<float> weights_;

    public:

    BinaryTree(int n) :
    num_leaves(n), current_node(n),
    parent_(2 * n - 1), children_(2 * (n - 1), -1), weights_(n - 1, -1.0f)
    {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int left_child(int n)
    {
        int i = n - num_leaves;
        return children_[2 * i];
    }

    int right_child(int n)
    {
        int i = n - num_leaves;
        return children_[2 * i + 1];
    }

    int parent(int n)
    {
        return parent_[n];
    }

    int add_node(int left_child, int right_child, float weight)
    {
        int p = current_node;

        parent_.at(p) = p;
        parent_.at(left_child) = p;
        parent_.at(right_child) = p;

        int i = p - num_leaves;
        children_.at(2 * i) = left_child;
        children_.at(2 * i + 1) = right_child;

        weights_.at(i) = weight;

        current_node++;
        return p;
    }
};