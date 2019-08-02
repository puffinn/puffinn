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
    void test_reset(DatasetDimensions dimensions, std::unique_ptr<HashSource<T>> source) {
        auto vec1 = UnitVectorFormat::generate_random(dimensions.actual);
        auto vec2 = UnitVectorFormat::generate_random(dimensions.actual);
        auto stored1 = to_stored_type<typename T::Format>(vec1, dimensions);
        auto stored2 = to_stored_type<typename T::Format>(vec2, dimensions);

        auto hasher = source->sample();
        source->reset(stored1.get());
        auto hash1 = (*hasher)();
        source->reset(stored2.get());
        auto hash2 = (*hasher)();
        REQUIRE(hash1 != hash2);
    }

    template <typename T>
    void test_hashes(
        DatasetDimensions dimensions,
        std::unique_ptr<HashSource<T>> source,
        unsigned int num_hashes,
        unsigned int hash_length
    ) {
        auto vec = UnitVectorFormat::generate_random(dimensions.actual);
        auto stored = to_stored_type<typename T::Format>(vec, dimensions);
        source->reset(stored.get());
        uint64_t max_hash = (((1 << (hash_length-1))-1) << 1)+1;

        std::vector<int> bit_occurences(hash_length, 0);
        for (unsigned int i=0; i < num_hashes; i++) {
            auto hasher = source->sample();
            uint64_t hash = (*hasher)();
            REQUIRE(hash <= max_hash);
            for (unsigned int bit=0; bit < hash_length; bit++) {
                if (hash & (1ull << bit)) {
                    bit_occurences[bit]++;
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
        auto dimensions = dataset.get_dimensions();
        test_reset<SimHash>(
            dimensions,
            HashPoolArgs<SimHash>(50).build(dimensions, 100, 0, 24));
        test_reset<FHTCrossPolytopeHash>(
            dimensions,
            HashPoolArgs<FHTCrossPolytopeHash>(80).build(dimensions, 100, 0, 20));
    }

    TEST_CASE("IndependentSource reset") {
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_dimensions();
        test_reset<SimHash>(
            dimensions,
            IndependentHashArgs<SimHash>().build(dimensions, 100, 2, 20));
        test_reset<FHTCrossPolytopeHash>(
            dimensions,
            IndependentHashArgs<FHTCrossPolytopeHash>().build(dimensions, 100, 2, 20));
    }

    TEST_CASE("TensoredHash reset") {
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_dimensions();
        test_reset<SimHash>(
            dimensions,
            TensoredHashArgs<SimHash>().build(dimensions, 100, 2, 20));
        test_reset<FHTCrossPolytopeHash>(
            dimensions,
            TensoredHashArgs<FHTCrossPolytopeHash>().build(dimensions, 100, 2, 20));
    }

    TEST_CASE("HashPool hashes") {
        const unsigned int HASH_LENGTH = 24;
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_dimensions();
        test_hashes<SimHash>(
            dimensions,
            HashPoolArgs<SimHash>(60).build(dimensions, 100, 0, HASH_LENGTH),
            100,
            HASH_LENGTH);
        test_hashes<FHTCrossPolytopeHash>(
            dimensions,
            HashPoolArgs<FHTCrossPolytopeHash>(60).build(dimensions, 100, 0, HASH_LENGTH),
            100,
            HASH_LENGTH);
    }

    TEST_CASE("HashPool sketches") {
        const unsigned int HASH_LENGTH = 64;
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_dimensions();
        test_hashes<SimHash>(
            dimensions,
            HashPoolArgs<SimHash>(60).build(dimensions, 100, 0, HASH_LENGTH),
            100,
            HASH_LENGTH);
    }


    TEST_CASE("Independent hashes") {
        const unsigned int HASH_LENGTH = 24;
        const unsigned int NUM_HASHES = 100;

        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_dimensions();
        test_hashes<SimHash>(
            dimensions,
            IndependentHashArgs<SimHash>().build(dimensions, 100, NUM_HASHES, HASH_LENGTH),
            NUM_HASHES,
            HASH_LENGTH);
        test_hashes<FHTCrossPolytopeHash>(
            dimensions,
            IndependentHashArgs<FHTCrossPolytopeHash>().build(dimensions, 100, NUM_HASHES, HASH_LENGTH),
            NUM_HASHES,
            HASH_LENGTH);
    }


    TEST_CASE("Tensored hashes") {
        const unsigned int HASH_LENGTH = 24;
        const unsigned int NUM_HASHES = 100;

        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_dimensions();
        test_hashes<SimHash>(
            dimensions,
            TensoredHashArgs<SimHash>().build(dimensions, 100, NUM_HASHES, HASH_LENGTH),
            NUM_HASHES,
            HASH_LENGTH);
        test_hashes<FHTCrossPolytopeHash>(
            dimensions,
            TensoredHashArgs<FHTCrossPolytopeHash>().build(dimensions, 100, NUM_HASHES, HASH_LENGTH),
            NUM_HASHES,
            HASH_LENGTH);
    }
}
