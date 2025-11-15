#ifndef UNION_FIND_H
#define UNION_FIND_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

/**
 * Tarjan's Union-Find (Disjoint Set Union) data structure
 * with path compression and union by rank.
 * Works with indices in the range [0, N-1].
 *
 * Time complexity: O(α(n)) amortized per operation, where α is the
 * inverse Ackermann function (practically constant for all reasonable n).
 */
class UnionFind {
private:
    std::vector<int> parent;     // parent[i] = parent of element i
    std::vector<int> rank;       // rank[i] = approximate depth of tree rooted at i
    std::vector<int> size;       // size[i] = size of component rooted at i
    int num_components;          // number of disjoint sets
    int num_nodes;

public:

    /**
     * Initialize empty union-find structure.
     */
    UnionFind() : num_components(0) {}

    /**
     * Initialize with n elements in the range [0, n-1].
     * Each element starts in its own set with size 1.
     * Time complexity: O(n)
     */
    explicit UnionFind(int n) :
    num_components(n), num_nodes(n),
    parent(n), rank(n, 0), size(n, 1) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    /**
     * Find the representative (root) of the set containing x.
     * Uses path compression for optimization.
     * Time complexity: O(α(n)) amortized
     */
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // path compression
        }
        return parent[x];
    }

    /**
     * Union the sets containing x and y.
     * Uses union by rank for optimization and updates component size.
     * Returns true if x and y were in different sets, false otherwise.
     * Time complexity: O(α(n)) amortized
     */
    int unite(int c_x, int c_y) {
        // Union by rank: attach smaller tree under root of deeper tree
        if (rank[c_x] > rank[c_y])
            std::swap(c_x, c_y);
        
        if (rank[c_x] == rank[c_y])
            rank[c_y]++;
        
        parent[c_x] = c_y;
        size[c_y] += size[c_x];

        num_nodes++;
        num_components--;

        return c_y;
    }

    /**
     * Check if x and y are in the same set.
     * Time complexity: O(α(n)) amortized
     */
    bool connected(int x, int y) {
        return find(x) == find(y);
    }

    /**
     * Get the size of the component containing x.
     * Time complexity: O(α(n)) amortized
     */
    int get_size(int x) {
        int root = find(x);
        return size[root];
    }

    /**
     * Check if component containing x meets size constraints.
     * Returns true if min_size <= component_size <= max_size.
     * Time complexity: O(α(n)) amortized
     */
    bool check_size(int x, int min_size, int max_size) {
        int comp_size = get_size(x);
        return comp_size >= min_size && comp_size <= max_size;
    }

    /**
     * Get the number of disjoint sets.
     * Time complexity: O(1)
     */
    int count() const {
        return num_components;
    }

    /**
     * Get the total number of elements.
     * Time complexity: O(1)
     */
    int total_elements() const {
        return parent.size();
    }

    /**
     * Get all root representatives.
     * Time complexity: O(n)
     */
    std::vector<int> get_roots() {
        std::vector<int> roots;
        roots.reserve(num_components);

        for (int i = 0; i < parent.size(); i++) {
            if (parent[i] == i) {
                roots.push_back(i);
            }
        }
        return roots;
    }

    /**
     * Get all elements in the component containing x.
     * Time complexity: O(n)
     */
    std::vector<int> get_component(int x) {
        int root = find(x);
        std::vector<int> component;
        component.reserve(size[root]);

        for (int i = 0; i < parent.size(); i++) {
            if (find(i) == root) {
                component.push_back(i);
            }
        }
        return component;
    }

    /**
     * Clear all data.
     * Time complexity: O(1) amortized (depends on deallocation)
     */
    void clear() {
        parent.clear();
        rank.clear();
        size.clear();
        num_components = 0;
    }
};

#endif // UNION_FIND_H
