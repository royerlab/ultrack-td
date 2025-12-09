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