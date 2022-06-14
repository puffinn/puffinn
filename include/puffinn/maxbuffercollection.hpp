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
    bool cmp_pair(const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
        return a.second > b.second
            || (a.second == b.second && a.first > b.first);
    }

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
        std::vector<ResultPair> data;

    public:

        MaxBufferCollection() 
            : n(0),
              k(0),
              capacity(0),
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
            this->capacity = k + 1;
            this->data.resize(n*(k+1));
        }

        float smallest_value(size_t idx) {
            return data[idx * capacity].second;
        }

        bool insert(size_t idx, size_t neighbor, float similarity) {
            similarity = std::min(1.0f, std::max(0.0f, similarity));
            size_t offset = idx * capacity;
            if (similarity <= data[offset].second) {
                return false;
            }

            for (size_t i=0; i<k; i++) {
                if (data[offset + i].first == neighbor) {
                    // insertion would be a duplicate, return true to signal that
                    // the point is not inserted because it's a duplicate
                    return true; 
                }
            }

            // get the slice holding data about `idx`
            auto begin = data.begin() + offset;
            auto end = begin + capacity;

            // write the new element at the end of the slice
            data[offset + k] = std::make_pair(neighbor, similarity);
            // rearrange the heap so that we pop the minimum similarity, placing it
            // on `end` to be overwritten by the next insertion
            std::push_heap(begin, end, cmp_pair);
            std::pop_heap(begin, end, cmp_pair);
            
            return true;
        }

        // Add all the best items stored in the `other` buffer
        void add_all(MaxBufferCollection & other) {
            assert(capacity == other.capacity);
            assert(n == other.n);
            #pragma omp parallel for
            for (size_t idx=0; idx<n; idx++) {
                size_t offset = capacity*idx;
                for (size_t i=0; i<k; i++) {
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
            std::vector<std::pair<uint32_t, float>> res;
            size_t offset = idx*capacity;
            for (size_t i=0; i<k; i++) {
                res.push_back(data[offset + i]);
            }
            std::sort(res.begin(), res.end(), cmp_pair);
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

    };
}
