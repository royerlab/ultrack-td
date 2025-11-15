#include <vector>
#include <unordered_map>

//     std::unordered_map<int, int> global_to_local;
//     int *local_to_global = new int[visited.size()];
// 
//     int *num_minima = new int[2 * visited.size() - 1];
//     std::memset(num_minima, 0, (2 * visited.size() - 1) * sizeof(int));
// 
//     float *mst_weights = new float[visited.size() - 1];
// 
//     for (int i = 0; i < visited.size(); i++) {
//         global_to_local[visited[i]] = i;
//         local_to_global[i] = visited[i];
//     }
//     int *local_edges = new int[edges.size()];
//     for (int i = 0; i < edges.size(); i++)
//         local_edges[i] = global_to_local[edges[i]];


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

    std::vector<int> apply_forward(const std::vector<int> &values) {
        std::vector<int> result(values.size());
        for (int i = 0; i < values.size(); i++) {
            result[i] = forward[values[i]];
        }
        return result;
    }
    
    std::vector<int> apply_backward(const std::vector<int> &values) {
        std::vector<int> result(values.size());
        for (int i = 0; i < values.size(); i++) {
            result[i] = backward[values[i]];
        }
        return result;
    }
};