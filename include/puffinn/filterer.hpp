#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/typedefs.hpp"
#include "puffinn/hash_source/deserialize.hpp"
#include "puffinn/hash_source/hash_source.hpp"
#include "puffinn/performance.hpp"

#include <cstring>
#include <immintrin.h>
#include <memory>

namespace puffinn {
    const size_t NUM_SKETCHES = 32;
    const size_t LOG_NUM_SKETCHES = 5;

    // Sketches for a single query.
    struct QuerySketches {
        // Sketches for the current query.
        std::vector<FilterLshDatatype> query_sketches;
        // Max hamming distance between sketches to be considered in the current query.
        uint_fast8_t max_sketch_diff;

        // Check if the value at position idx in the dataset passes the next filter.
        // A value can only pass one filter.
        bool passes_filter(FilterLshDatatype sketch, int_fast32_t sketch_idx) const {
            uint_fast8_t sketch_diff = popcountll(sketch ^ query_sketches[sketch_idx]);
            return (sketch_diff <= max_sketch_diff);
        }
    };

    template <typename T>
    class Filterer {
        std::unique_ptr<HashSource<T>> hash_source;
        // Filter hash functions
        std::vector<std::unique_ptr<Hash>> hash_functions;

        // Filters are stored with sketches for the same value adjacent.
        std::vector<FilterLshDatatype> sketches;
        std::unique_ptr<HashSourceArgs<T>> sketch_args;

    public:
        Filterer(const HashSourceArgs<T>& args, DatasetDescription<typename T::Sim::Format> dataset)
          : hash_source(
                args.build(
                    dataset,
                    NUM_SKETCHES,
                    NUM_FILTER_HASHBITS)),
            sketch_args(args.copy())
        {
            for (size_t i=0; i<NUM_SKETCHES; i++) {
                hash_functions.push_back(hash_source->sample());
            }
        }

        Filterer(std::istream& in) {
            sketch_args = deserialize_hash_args<T>(in);
            hash_source = sketch_args->deserialize_source(in);
            hash_functions.reserve(NUM_SKETCHES);
            for (size_t i=0; i < NUM_SKETCHES; i++) {
                hash_functions.push_back(hash_source->deserialize_hash(in));
            }
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
            sketches.resize(len);
            in.read(reinterpret_cast<char*>(sketches.data()), len*sizeof(FilterLshDatatype));
        }

        void serialize(std::ostream& out) const {
            sketch_args->serialize(out);
            hash_source->serialize(out);
            for (auto& h : hash_functions) {
                h->serialize(out);
            }
            size_t len = sketches.size();
            out.write(reinterpret_cast<char*>(&len), sizeof(size_t));
            out.write(reinterpret_cast<const char*>(sketches.data()), len*sizeof(FilterLshDatatype));
        }

        uint64_t memory_usage(DatasetDescription<typename T::Sim::Format> dataset) {
            return sketch_args->memory_usage(dataset, NUM_SKETCHES, NUM_FILTER_HASHBITS)
                + sketches.size()*sizeof(FilterLshDatatype)
                + NUM_SKETCHES*sketch_args->function_memory_usage(dataset, NUM_FILTER_HASHBITS);
        }

        void add_sketches(
            const Dataset<typename T::Sim::Format>& dataset,
            uint32_t first_index
        ) {
            sketches.reserve(dataset.get_size()*NUM_SKETCHES);

            for (size_t idx = first_index; idx < dataset.get_size(); idx++) {
                auto state = hash_source->reset(dataset[idx]);
                for (size_t sketch_index = 0; sketch_index < NUM_SKETCHES; sketch_index++) {
                    sketches.push_back((*hash_functions[sketch_index])(state.get()));
                }
            }
        }

        QuerySketches reset(typename T::Sim::Format::Type* vec) const {
            auto state = hash_source->reset(vec);

            QuerySketches res;
            res.query_sketches.reserve(NUM_SKETCHES);
            for (size_t sketch_index=0; sketch_index<NUM_SKETCHES; sketch_index++) {
                res.query_sketches.push_back((*hash_functions[sketch_index])(state.get()));
            }
            res.max_sketch_diff = NUM_FILTER_HASHBITS;
            return res;
        }

        void prefetch(uint32_t idx, int_fast32_t sketch_idx) const {
            prefetch_addr(&sketches[(idx << LOG_NUM_SKETCHES) | sketch_idx]);
        }

        uint_fast8_t get_max_sketch_diff(float min_dist) const {
            float collision_prob = hash_source->collision_probability(min_dist, 1);
            return (NUM_FILTER_HASHBITS*(1.0-collision_prob));
        }

        FilterLshDatatype get_sketch(uint32_t idx, int_fast32_t sketch_idx) const {
            return sketches[(idx << LOG_NUM_SKETCHES) | sketch_idx];
        }
    };
}
