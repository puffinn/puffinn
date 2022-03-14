#pragma once

#include "puffinn/typedefs.hpp"
#include "puffinn/performance.hpp"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

namespace puffinn {
    // Stores the `k` indices with the highest similarities seen so far. Similarities are always values between 0 and 1. 
    class MaxPairBuffer {
    public:
        using ResultPair = std::pair<std::pair<uint32_t, uint32_t>, float>;

    private:
        const unsigned int size;
        unsigned int inserted_values;
        float minval;
        std::vector<ResultPair> data;

        // Reorder the values, so that the top `k` elements are stored first.
        // All other values are removed.
        void filter() {
            g_performance_metrics.start_timer(Computation::MaxbufferFilter);
            std::sort(data.begin(), data.begin()+inserted_values,
                [](const ResultPair& a, const ResultPair& b) {
                    return a.second > b.second
                        || (a.second == b.second && a.first > b.first);
                });

            // Deduplication step
            unsigned int deduplicated_values = std::min(1u, inserted_values);
            for (unsigned int idx=1; idx < inserted_values; idx++) {
                if (data[idx].first != data[deduplicated_values-1].first) {
                    data[deduplicated_values] = data[idx];
                    deduplicated_values++;
                }
            }
            inserted_values = std::min(deduplicated_values, size);
            if (inserted_values == size && size != 0) {
                minval = data[inserted_values-1].second;
            }
            g_performance_metrics.store_time(Computation::MaxbufferFilter);
        }

    public:
        // Construct a buffer containing `size` elements. The memory used is twice that.
        MaxPairBuffer(unsigned int k)
          : size(k),
            inserted_values(0),
            minval(0.0),
            data(std::vector<ResultPair>(2*k))
        {
            if (k == 0) {
                // Make it impossible to insert.
                minval = 1.0;
            }
        }

        // Insert an index with an associated value into the buffer.
        // The buffer may choose to ignore it if it is not relevant.
        bool insert(std::pair<uint32_t, uint32_t> idx, float value) {
            value = std::min(1.0f, std::max(0.0f, value));
            // Value is not relevant
            // This will also discard points with similarity=0, but this will not have an effect on the result.
            if (value <= minval) { return false; }

            if (inserted_values == 2*size) {
                filter();
            }
            auto elements = idx;
            if (idx.first > idx.second) {
                elements = {idx.second, idx.first};
            }
            data[inserted_values] = { elements, value };
            inserted_values++;
            return true;
        }

        // Retrieve the `k` entries with the highest associated values.
        std::vector<ResultPair> best_entries() {
            filter();
            std::vector<ResultPair> res;
            for (unsigned i=0; i<inserted_values; i++) {
                res.push_back(data[i]);
            }
            return res;
        }

        std::vector<std::pair<uint32_t, uint32_t>> best_indices() {
            auto entries = best_entries();
            std::vector<std::pair<uint32_t, uint32_t>> res;
            res.reserve(entries.size());
            for (auto entry : entries) {
                res.push_back(entry.first);
            }
            return res;
        }

        // Retrieve the current smallest values that inserted values have to beat
        // in order to be considered.
        float smallest_value() const {
            return minval;
        }

        // Retrieve the i'th inserted index when sorted by their descending values.
        //
        // buffer[0] will be the index with the largest associated value,
        // buffer[k-1] will be the index with the least associated value.
        // This does not take into account values that are inserted after the last filter.
        std::pair<uint32_t, uint32_t> operator[](size_t i) const {
            return data[i].first;
        }
    };
}
