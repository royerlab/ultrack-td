#pragma once

#include <vector>
#include <list>
#include <numeric>
#include <algorithm>

/**
 * Tarjan's Union-Find (Disjoint Set Union) data structure
 * with path compression and union by rank.
 * Works with indices in the range [0, N-1].
 *
 * Template parameter TrackComponents:
 * - false (default): Standard union-find, get_component() is O(n)
 * - true: Maintains component lists, get_component() is O(1) but uses more memory
 *
 * Time complexity: O(α(n)) amortized per operation, where α is the
 * inverse Ackermann function (practically constant for all reasonable n).
 */
template<bool TrackComponents = false>
class UnionFind {
private:
    std::vector<int> parent;     // parent[i] = parent of element i
    std::vector<int> rank;       // rank[i] = approximate depth of tree rooted at i
    std::vector<int> size;       // size[i] = size of component rooted at i
    int num_components;          // number of disjoint sets
    int num_nodes;

    // Only instantiated when TrackComponents = true
    std::vector<std::list<int>> components;  // components[root] = list of all members

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

        if constexpr (TrackComponents) {
            components.resize(n);
            for (int i = 0; i < n; i++) {
                components[i].push_back(i);
            }
        }
    }

    /**
     * Find the representative (root) of the set containing x.
     * Uses path halving - single loop that compresses the path while finding root.
     * Time complexity: O(α(n)) amortized
     */
    inline int find(int x) noexcept {
        // Path halving: make every other node point to its grandparent
        int root = x;
        while (parent[root] != root) {
            root = parent[root];
        }

        // Path compression: make every node point to the root
        while (x != root) {
            int next = parent[x];
            parent[x] = root;
            x = next;
        }
        return x;
    }

    /**
     * Union the sets containing x and y.
     * Uses union by rank for optimization and updates component size.
     * Returns the new root of the merged component.
     * Time complexity: O(α(n)) amortized, O(1) if TrackComponents=true (list splice)
     */
    inline int unite(int c_x, int c_y) noexcept {
        // Union by rank: attach smaller tree under root of deeper tree
        if (rank[c_x] > rank[c_y])
            std::swap(c_x, c_y);

        if (rank[c_x] == rank[c_y])
            rank[c_y]++;

        parent[c_x] = c_y;
        size[c_y] += size[c_x];

        if constexpr (TrackComponents) {
            // Merge component lists: O(1) splice operation
            components[c_y].splice(components[c_y].end(), components[c_x]);
        }

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
    inline int get_size(int x) noexcept {
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
     * Get const reference to the component list containing x.
     * Only available when TrackComponents=true.
     * Time complexity: O(α(n)) for find
     */
    template<bool T = TrackComponents>
    typename std::enable_if<T, const std::list<int>&>::type
    get_component_list(int x) {
        int root = find(x);
        return components[root];
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

        if constexpr (TrackComponents) {
            components.clear();
        }
    }
};
