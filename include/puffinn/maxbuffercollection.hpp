#pragma once

#include "puffinn/typedefs.hpp"
#include "puffinn/performance.hpp"
#include <cassert>
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
        size_t n;
        size_t k;
        size_t capacity;
        std::vector<size_t> write_head;
        std::vector<float> minval;
        std::vector<ResultPair> data;

    public:

        MaxBufferCollection() 
            : n(0),
              k(0),
              capacity(0),
              write_head(0),
              minval(0),
              data(0)
        {
            // this constructor is meant only for pre-allocating space when building vectors of MaxBufferCollections
        }

        void init(size_t n, size_t k)
        { 
            if (k == 0) {
                throw std::invalid_argument("k should be > 0");
            }
            this->n = n;
            this->k = k;
            this->capacity = 2*k;
            this->write_head.resize(n);
            this->minval.resize(n);
            this->data.resize(2*k*n);
        }

        float smallest_value(size_t idx) {
            return minval[idx];
        }

        bool insert(size_t idx, size_t neighbor, float similarity) {
            similarity = std::min(1.0f, std::max(0.0f, similarity));
            if (similarity <= minval[idx]) {
                return false;
            }

            size_t offset = idx * capacity;
            for (size_t i=0; i<write_head[idx]; i++) {
                if (data[offset + i].first == neighbor) {
                    // insertion would be a duplicate, return true to signal that
                    // the point is not discarded because it's a duplicate
                    return true; 
                }
            }
            
            if (write_head[idx] == capacity) {
                filter(idx);
            }

            data[offset + write_head[idx]] = std::make_pair(neighbor, similarity);
            write_head[idx]++;

            return true;
        }

        // Add all the best items stored in the `other` buffer
        void add_all(MaxBufferCollection & other) {
            assert(capacity == other.capacity);
            assert(n == other.n);
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
            std::sort(data.begin() + offset, data.begin() + offset + write_head[idx], 
                [](const ResultPair& a, const ResultPair& b) {
                    return a.second > b.second
                        || (a.second == b.second && a.first > b.first);
                });

            // update the write head, effectively removing all the elements 
            // with similarity smaller than minval[idx]
            write_head[idx] = std::min(write_head[idx], k);

            // update minval
            if (write_head[idx] == k) {
                minval[idx] = data[offset + k - 1].second;
            }
        }

    };
}
