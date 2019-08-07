#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/typedefs.hpp"
#include "puffinn/hash_source/hash_source.hpp"
#include "puffinn/performance.hpp"

#include <cstring>
#include <immintrin.h>
#include <memory>

namespace puffinn {
    const size_t NUM_SKETCHES = 32;
    const size_t LOG_NUM_SKETCHES = 5;

    template <typename T>
    class Filterer {
        std::unique_ptr<HashSource<T>> hash_source;
        // Filter hash functions
        std::vector<std::unique_ptr<Hash>> hash_functions;

        // Filters are stored with sketches for the same value adjacent.
        std::vector<FilterLshDatatype> sketches;

        // Sketches for the current query.
        std::vector<FilterLshDatatype> query_sketches;
        // Max hamming distance between sketches to be considered in the current query.
        uint_fast8_t max_sketch_diff;

    public:
        Filterer(std::unique_ptr<HashSource<T>> source)
          : hash_source(std::move(source))
        {
            for (size_t i=0; i<NUM_SKETCHES; i++) {
                hash_functions.push_back(hash_source->sample());
            }
        }

        void add_sketches(
            const Dataset<typename T::Sim::Format>& dataset,
            uint32_t first_index
        ) {
            sketches.reserve(dataset.get_size()*NUM_SKETCHES);

            for (size_t idx = first_index; idx < dataset.get_size(); idx++) {
                hash_source->reset(dataset[idx]);
                for (size_t sketch_index = 0; sketch_index < NUM_SKETCHES; sketch_index++) {
                    sketches.push_back((*hash_functions[sketch_index])());
                }
            }
        }

        void reset(typename T::Sim::Format::Type* vec) {
            hash_source->reset(vec);
            query_sketches.clear();
            for (size_t sketch_index=0; sketch_index<NUM_SKETCHES; sketch_index++) {
                query_sketches.push_back((*hash_functions[sketch_index])());
            }
            max_sketch_diff = NUM_FILTER_HASHBITS;
        }

        void prefetch(uint32_t idx, int_fast32_t sketch_idx) {
            __builtin_prefetch(&sketches[(idx << LOG_NUM_SKETCHES) | sketch_idx]);
        }

        void update_max_sketch_diff(float min_dist) {
            float collision_prob = hash_source->collision_probability(min_dist, 1);
            max_sketch_diff = (uint_fast8_t) (NUM_FILTER_HASHBITS*(1.0-collision_prob));
        }
/*
        bool passes_filter2(
            uint32_t idx1, uint32_t idx2, uint32_t idx3, uint32_t idx4,
            int_fast32_t sketch_idx
        ) const {
            auto query_sketch = _mm256_set1_epi64x(query_sketches[sketch_idx]);
            auto res = _mm256_set_epi64x(
                sketches[(idx1 << LOG_NUM_SKETCHES) | sketch_idx],
                sketches[(idx2 << LOG_NUM_SKETCHES) | sketch_idx],
                sketches[(idx3 << LOG_NUM_SKETCHES) | sketch_idx],
                sketches[(idx4 << LOG_NUM_SKETCHES) | sketch_idx]);
            res = _mm256_xor_si256(res, query_sketch);
            res = _mm256_popcnt_epi32(res);
            (*alignas(32) FilterLshDatatype sketches[] = {
                sketches[(idx1 << LOG_NUM_SKETCHES) | sketch_idx],
                sketches[(idx2 << LOG_NUM_SKETCHES) | sketch_idx],
                sketches[(idx3 << LOG_NUM_SKETCHES) | sketch_idx],
                sketches[(idx4 << LOG_NUM_SKETCHES) | sketch_idx]
            };*)
            return true;
        }
*/
        // Check if the value at position idx in the dataset passes the next filter.
        // A value can only pass one filter.
        bool passes_filter(uint32_t idx, int_fast32_t sketch_idx) {
            auto sketch = sketches[(idx << LOG_NUM_SKETCHES) | sketch_idx];
            uint_fast8_t sketch_diff = __builtin_popcountll(sketch ^ query_sketches[sketch_idx]);
            return (sketch_diff <= max_sketch_diff);
        }

        FilterLshDatatype get_sketch(uint32_t idx, int_fast32_t sketch_idx) {
            return sketches[(idx << LOG_NUM_SKETCHES) | sketch_idx];
        }
    };
}
