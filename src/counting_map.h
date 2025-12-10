#include <unordered_map>


template <typename Key, typename Value = long>
class CountingMap {
public:
    using map_type = std::unordered_map<Key, Value>;

    // start counting at `offset` (default 0)
    explicit CountingMap(Value offset = 0)
        : next_value_(offset) {}

    // Query: returns existing value, or assigns & returns next_value_
    Value get(const Key& key) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            return it->second;
        }
        Value v = next_value_++;
        map_.emplace(key, v);
        return v;
    }

    Value next_value() const {
        return next_value_;
    }

private:
    map_type map_;
    Value next_value_;
};
