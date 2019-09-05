#pragma once

#include "catch.hpp"
#include "puffinn/filterer.hpp"
#include "puffinn/format/unit_vector.hpp"
#include "puffinn/hash_source/independent.hpp"
#include "puffinn/hash/simhash.hpp"

using namespace puffinn;

namespace filterer_test {
    TEST_CASE("filtering equal/opposite vector") {
        Dataset<UnitVectorFormat> dataset(2);
        dataset.insert(std::vector<float>({1, 0}));
        dataset.insert(std::vector<float>({-1, 0}));

        IndependentHashArgs<SimHash> hash_args;
        Filterer<SimHash> filterer(hash_args, dataset.get_description());

        filterer.add_sketches(dataset, 0);

        std::vector<float> query({1, 0});
        auto stored = to_stored_type<UnitVectorFormat>(query, dataset.get_description());
        auto sketches = filterer.reset(stored.get());

        // Anything initially passes
        for (size_t i=0; i < NUM_SKETCHES; i++) {
            REQUIRE(sketches.passes_filter(filterer.get_sketch(1, i), i));
        }

        sketches.max_sketch_diff = filterer.get_max_sketch_diff(1.0);
        for (size_t i=0; i < NUM_SKETCHES; i++) {
            REQUIRE(sketches.passes_filter(filterer.get_sketch(0, i), i));
            REQUIRE(!sketches.passes_filter(filterer.get_sketch(1, i), i));
        }

        sketches.max_sketch_diff = filterer.get_max_sketch_diff(0.0);
        for (size_t i=0; i < NUM_SKETCHES; i++) {
            REQUIRE(sketches.passes_filter(filterer.get_sketch(1, i), i));
        }
    }

    TEST_CASE("All filtering bits used") {
        const int NUM_VECTORS = 20;
        const unsigned int DIMENSIONS = 100;

        Dataset<UnitVectorFormat> dataset(DIMENSIONS);
        for (int i=0; i < NUM_VECTORS; i++) {
            dataset.insert(UnitVectorFormat::generate_random(DIMENSIONS));
        }

        IndependentHashArgs<SimHash> hash_args;
        Filterer<SimHash> filterer(hash_args, dataset.get_description());
        filterer.add_sketches(dataset, 0);

        int bit_counts[NUM_FILTER_HASHBITS];
        for (int idx=0; idx < NUM_VECTORS; idx++) {
            for (unsigned int sketch=0; sketch < NUM_SKETCHES; sketch++) {
                for (unsigned int bit=0; bit < NUM_FILTER_HASHBITS; bit++) {
                    if (filterer.get_sketch(idx, sketch) & (1 << bit)) {
                        bit_counts[bit]++;
                    }
                }
            }
        }
        for (unsigned int bit=0; bit < NUM_FILTER_HASHBITS; bit++) {
            REQUIRE(bit_counts[bit] != 0);
        }
    }
}
