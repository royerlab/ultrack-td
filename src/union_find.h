#ifndef UNION_FIND_H
#define UNION_FIND_H

#include <vector>
#include <unordered_map>
#include <algorithm>

/**
 * Tarjan's Union-Find (Disjoint Set Union) data structure
 * with path compression and union by rank.
 * Supports arbitrary indices and tracks component sizes.
 *
 * Uses a hybrid approach: maps arbitrary indices to a dense array internally
 * for O(1) access while supporting sparse index sets.
 *
 * Time complexity: O(α(n)) amortized per operation, where α is the
 * inverse Ackermann function (practically constant for all reasonable n).
 */
class UnionFind {
private:
    std::vector<int> parent;              // parent[i] = parent of element i
    std::vector<int> rank;                // rank[i] = approximate depth of tree rooted at i
    std::vector<int> size;                // size[i] = size of component rooted at i
    std::unordered_map<int, int> id_map;  // maps external index -> internal index
    std::vector<int> reverse_map;          // maps internal index -> external index
    int num_components;                    // number of disjoint sets

public:
    /**
     * Initialize empty union-find structure.
     */
    UnionFind() : num_components(0) {}

    /**
     * Initialize with a list of elements (e.g., visited array).
     * Each element starts in its own set with size 1.
     * Time complexity: O(n)
     */
    explicit UnionFind(const std::vector<int>& elements) : num_components(0) {
        int n = elements.size();
        parent.reserve(n);
        rank.reserve(n);
        size.reserve(n);
        reverse_map.reserve(n);
        id_map.reserve(n);

        for (int elem : elements) {
            add(elem);
        }
    }

    /**
     * Add a new element to the structure.
     * If element already exists, does nothing.
     * Time complexity: O(1) average
     */
    void add(int x) {
        if (id_map.find(x) == id_map.end()) {
            int internal_id = parent.size();
            id_map[x] = internal_id;
            reverse_map.push_back(x);
            parent.push_back(internal_id);
            rank.push_back(0);
            size.push_back(1);
            num_components++;
        }
    }

    /**
     * Check if element exists.
     * Time complexity: O(1) average
     */
    bool contains(int x) const {
        return id_map.find(x) != id_map.end();
    }

    /**
     * Find the representative (root) of the set containing x.
     * Uses path compression for optimization.
     * Automatically adds element if it doesn't exist.
     * Time complexity: O(α(n)) amortized
     */
    int find(int x) {
        // Add element if it doesn't exist
        if (id_map.find(x) == id_map.end()) {
            add(x);
        }

        int internal_id = id_map[x];
        return find_internal(internal_id);
    }

    /**
     * Union the sets containing x and y.
     * Uses union by rank for optimization and updates component size.
     * Returns true if x and y were in different sets, false otherwise.
     * Time complexity: O(α(n)) amortized
     */
    bool unite(int x, int y) {
        int id_x = get_or_add_id(x);
        int id_y = get_or_add_id(y);

        int root_x = find_internal(id_x);
        int root_y = find_internal(id_y);

        if (root_x == root_y) {
            return false;  // already in the same set
        }

        // Union by rank: attach smaller tree under root of deeper tree
        if (rank[root_x] < rank[root_y]) {
            parent[root_x] = root_y;
            size[root_y] += size[root_x];
        } else if (rank[root_x] > rank[root_y]) {
            parent[root_y] = root_x;
            size[root_x] += size[root_y];
        } else {
            parent[root_y] = root_x;
            size[root_x] += size[root_y];
            rank[root_x]++;
        }

        num_components--;
        return true;
    }

    /**
     * Check if x and y are in the same set.
     * Time complexity: O(α(n)) amortized
     */
    bool connected(int x, int y) {
        if (!contains(x) || !contains(y)) {
            return false;
        }
        return find_internal(id_map[x]) == find_internal(id_map[y]);
    }

    /**
     * Get the size of the component containing x.
     * Time complexity: O(α(n)) amortized
     */
    int get_size(int x) {
        if (!contains(x)) {
            return 0;
        }
        int root = find_internal(id_map[x]);
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
     * Get all root representatives (external indices).
     * Time complexity: O(n)
     */
    std::vector<int> get_roots() {
        std::vector<int> roots;
        roots.reserve(num_components);

        for (int i = 0; i < parent.size(); i++) {
            if (parent[i] == i) {
                roots.push_back(reverse_map[i]);
            }
        }
        return roots;
    }

    /**
     * Get all elements in the component containing x (external indices).
     * Time complexity: O(n)
     */
    std::vector<int> get_component(int x) {
        if (!contains(x)) {
            return {};
        }

        int root = find_internal(id_map[x]);
        std::vector<int> component;
        component.reserve(size[root]);

        for (int i = 0; i < parent.size(); i++) {
            if (find_internal(i) == root) {
                component.push_back(reverse_map[i]);
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
        id_map.clear();
        reverse_map.clear();
        num_components = 0;
    }

private:
    /**
     * Internal find using dense array indices.
     * Time complexity: O(α(n)) amortized
     */
    int find_internal(int internal_id) {
        if (parent[internal_id] != internal_id) {
            parent[internal_id] = find_internal(parent[internal_id]);  // path compression
        }
        return parent[internal_id];
    }

    /**
     * Get internal ID for external index, adding if necessary.
     * Time complexity: O(1) average
     */
    int get_or_add_id(int x) {
        auto it = id_map.find(x);
        if (it != id_map.end()) {
            return it->second;
        }
        add(x);
        return id_map[x];
    }
};

#endif // UNION_FIND_H
