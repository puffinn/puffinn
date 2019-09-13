#pragma once

#include "catch.hpp"
#include "puffinn/hash_source/pool.hpp"
#include "puffinn/hash_source/independent.hpp"
#include "puffinn/hash_source/tensor.hpp"
#include "puffinn/hash/simhash.hpp"
#include "puffinn/hash/crosspolytope.hpp"

using namespace puffinn;

namespace hash_source {
    template <typename T>
    void test_reset(DatasetDescription<typename T::Sim::Format> dimensions, std::unique_ptr<HashSource<T>> source) {
        auto vec1 = UnitVectorFormat::generate_random(dimensions.args);
        auto vec2 = UnitVectorFormat::generate_random(dimensions.args);
        auto stored1 = to_stored_type<typename T::Sim::Format>(vec1, dimensions);
        auto stored2 = to_stored_type<typename T::Sim::Format>(vec2, dimensions);

        auto hasher = source->sample();
        auto state = source->reset(stored1.get());
        auto hash1 = (*hasher)(state.get());
        state = source->reset(stored2.get());
        auto hash2 = (*hasher)(state.get());
        REQUIRE(hash1 != hash2);
    }

    template <typename T>
    void test_hashes(
        DatasetDescription<typename T::Sim::Format> dimensions,
        std::unique_ptr<HashSource<T>> source,
        unsigned int num_hashes,
        unsigned int hash_length
    ) {
        std::vector<int> bit_occurences(hash_length, 0);
        std::vector<std::unique_ptr<Hash>> hash_functions;
        for (unsigned int i=0; i < num_hashes; i++) {
            hash_functions.push_back(source->sample());
        }

        // Test with a couple of different vectors since the limited range of some hash functions
        // (such as FHTCrossPolytope) can lead to some bits unused.
        for (int vec_idx = 0; vec_idx < 2; vec_idx++) {
            auto vec = UnitVectorFormat::generate_random(dimensions.args);
            auto stored = to_stored_type<typename T::Sim::Format>(vec, dimensions);
            auto state = source->reset(stored.get());
            uint64_t max_hash = (((1llu << (hash_length-1))-1) << 1)+1;

            for (unsigned int i=0; i < num_hashes; i++) {
                uint64_t hash = (*hash_functions[i])(state.get());
                REQUIRE(hash <= max_hash);
                for (unsigned int bit=0; bit < hash_length; bit++) {
                    if (hash & (1ull << bit)) {
                        bit_occurences[bit]++;
                    }
                }
            }
        }
        for (unsigned int bit=0; bit < hash_length; bit++) {
            // All bits used.
            REQUIRE(bit_occurences[bit] > 0);
        }
    }

    TEST_CASE("HashPool reset") {
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_description();
        test_reset<SimHash>(
            dimensions,
            HashPoolArgs<SimHash>(50).build(dimensions, 0, 24));
        test_reset<FHTCrossPolytopeHash>(
            dimensions,
            HashPoolArgs<FHTCrossPolytopeHash>(80).build(dimensions, 0, 20));
    }

    TEST_CASE("IndependentSource reset") {
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_description();
        test_reset<SimHash>(
            dimensions,
            IndependentHashArgs<SimHash>().build(dimensions, 2, 20));
        test_reset<FHTCrossPolytopeHash>(
            dimensions,
            IndependentHashArgs<FHTCrossPolytopeHash>().build(dimensions, 2, 20));
    }

    TEST_CASE("TensoredHash reset") {
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_description();
        test_reset<SimHash>(
            dimensions,
            TensoredHashArgs<SimHash>().build(dimensions, 2, 20));
        test_reset<FHTCrossPolytopeHash>(
            dimensions,
            TensoredHashArgs<FHTCrossPolytopeHash>().build(dimensions, 2, 20));
    }

    TEST_CASE("HashPool hashes") {
        const unsigned int HASH_LENGTH = 24;
        unsigned int samples = 100;
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_description();
        test_hashes<SimHash>(
            dimensions,
            HashPoolArgs<SimHash>(60).build(dimensions, samples, HASH_LENGTH),
            samples,
            HASH_LENGTH);
        test_hashes<FHTCrossPolytopeHash>(
            dimensions,
            HashPoolArgs<FHTCrossPolytopeHash>(60).build(dimensions, samples, HASH_LENGTH),
            samples,
            HASH_LENGTH);
    }

    TEST_CASE("HashPool sketches") {
        const unsigned int HASH_LENGTH = 64;
        unsigned int samples = 100;
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_description();
        test_hashes<SimHash>(
            dimensions,
            HashPoolArgs<SimHash>(60).build(dimensions, samples, HASH_LENGTH),
            samples,
            HASH_LENGTH);
    }


    TEST_CASE("Independent hashes") {
        const unsigned int HASH_LENGTH = 24;
        const unsigned int NUM_HASHES = 100;

        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_description();
        test_hashes<SimHash>(
            dimensions,
            IndependentHashArgs<SimHash>().build(dimensions, NUM_HASHES, HASH_LENGTH),
            NUM_HASHES,
            HASH_LENGTH);
        test_hashes<FHTCrossPolytopeHash>(
            dimensions,
            IndependentHashArgs<FHTCrossPolytopeHash>().build(dimensions, NUM_HASHES, HASH_LENGTH),
            NUM_HASHES,
            HASH_LENGTH);
    }


    TEST_CASE("Tensored hashes") {
        const unsigned int HASH_LENGTH = 24;
        const unsigned int NUM_HASHES = 100;

        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_description();
        test_hashes<SimHash>(
            dimensions,
            TensoredHashArgs<SimHash>().build(dimensions, NUM_HASHES, HASH_LENGTH),
            NUM_HASHES,
            HASH_LENGTH);
        test_hashes<FHTCrossPolytopeHash>(
            dimensions,
            TensoredHashArgs<FHTCrossPolytopeHash>().build(dimensions, NUM_HASHES, HASH_LENGTH),
            NUM_HASHES,
            HASH_LENGTH);
    }
}
