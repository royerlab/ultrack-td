#include <vector>
#include <numeric>


struct BinaryTree {

    private:

    int current_node;
    std::vector<int> parent_;
    std::vector<int> children_;
    std::vector<float> weight_;

    public:

    int num_leaves;

    BinaryTree(int n) :
    num_leaves(n), current_node(n),
    parent_(2 * n - 1), children_(2 * (n - 1), -1), weight_(n - 1, -1.0f)
    {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int left_child(int n) const noexcept
    {
        int i = n - num_leaves;
        return children_[2 * i];
    }

    int right_child(int n) const noexcept
    {
        int i = n - num_leaves;
        return children_[2 * i + 1];
    }

    int parent(int n) const noexcept
    {
        return parent_[n];
    }

    // FIXME: critical, this should be float but breaks the code
    int weight(int n) noexcept
    {
        return weight_[n - num_leaves];
    }

    inline int add_node(int left_child, int right_child, float weight) noexcept
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