#pragma once

#include "puffinn/typedefs.hpp"
#include "puffinn/performance.hpp"

#include <iostream>
#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

namespace puffinn {
    // This class is functionally equivalent to std::vector<MaxBuffer>.
    // The advantage is that it puts less pressure on the memory allocator,
    // thus taking far less time to initialize. Furthermore, it is more cache friendly,
    // since there is less indirection.
    class MaxBufferCollection {
    public:
        using ResultPair = std::pair<uint32_t, float>;
    private:
        const size_t n;
        const size_t k;
        const size_t capacity;
        // This array of size `capacity` will be used to sort, indirectly, chunks 
        // of the arrays similarities and indices
        std::vector<size_t> write_head;
        std::vector<float> minval;
        std::vector<ResultPair> data;

    public:

        MaxBufferCollection(size_t n, size_t k)
            : n(n),
              k(k),
              capacity(2*k),
              write_head(n),
              minval(n),
              data(2*k*n)
        { 
            if (k == 0) {
                throw std::invalid_argument("k should be > 0");
            }
        }

        // return the the k-th smallest value, if k elements 
        // have been inserted, otherwise return a negative value
        float kth_value(size_t idx) {
            if (write_head[idx] >= k) {
                filter(idx);
                return data[idx*capacity + k - 1].second;
            } else {
                return -std::numeric_limits<float>::infinity();
            }
        }

        bool insert(size_t idx, size_t neighbor, float similarity) {
            if (similarity <= minval[idx]) {
                return false;
            }

            if (write_head[idx] == capacity) {
                filter(idx);
            }

            size_t offset = idx * capacity;
            data[offset + write_head[idx]] = std::make_pair(neighbor, similarity);
            write_head[idx]++;

            return true;
        }

        // Add all the best items stored in the `other` buffer
        void add_all(MaxBufferCollection & other) {
            #pragma omp parallel for
            for (size_t idx=0; idx<n; idx++) {
                other.filter(idx);
                size_t offset = capacity*idx;
                for (size_t i=0; i<other.write_head[idx]; i++) {
                    auto & o = other.data[offset+i];
                    if ( !insert(idx, o.first, o.second) ) {
                        // we stop inserting as soon as we hit the first
                        // value smaller than the minimum of `this`, since the
                        // iteration on `other` is by decreasing values
                        break;
                    }
                }
            }
        }

        std::vector<ResultPair> best_entries(size_t idx) {
            if (write_head[idx] >= k) {
                filter(idx);
            }
            std::vector<std::pair<uint32_t, float>> res;
            size_t offset = idx*capacity;
            size_t end = std::min(write_head[idx], k);
            for (size_t i=0; i<end; i++) {
                res.push_back(data[offset + i]);
            }
            // std::cout << "--------------" << std::endl;
            // PRINTVEC(res);
            // std::sort(res.begin(), res.end(),
            //     [](const ResultPair& a, const ResultPair& b) {
            //         return a.second > b.second
            //             || (a.second == b.second && a.first > b.first);
            //     });
            return res;
        }

        std::vector<uint32_t> best_indices(size_t idx) {
            auto entries = best_entries(idx);
            std::vector<uint32_t> res;
            res.reserve(entries.size());
            for (auto entry : entries) {
                res.push_back(entry.first);
            }
            return res;
        }

    private:
        void filter(size_t idx) {
            size_t offset = idx * capacity;
            // sort the indices buffer
            std::sort(data.begin() + offset, data.begin() + offset + capacity, 
                [](const ResultPair& a, const ResultPair& b) {
                    return a.second > b.second
                        || (a.second == b.second && a.first > b.first);
                });

            size_t deduplicated_values = std::min(1ul, write_head[idx]);
            for (size_t i=1; i < write_head[idx]; i++) {
                if (data[offset + i].first != data[offset + deduplicated_values-1].first) {
                    data[offset + deduplicated_values] = data[offset + i];
                    deduplicated_values++;
                }
            }
            write_head[idx] = std::min(deduplicated_values, k);

            // update minval
            if (write_head[idx] == k) {
                minval[idx] = data[offset + k - 1].second;
            }

            // update the write head, effectively removing all the elements 
            // with similarity smaller than minval[idx]
            write_head[idx] = k;
        }

    };
}
