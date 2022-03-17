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
        auto state = source->reset(stored1.get(), false);
        auto hash1 = (*hasher)(state.get());
        state = source->reset(stored2.get(), false);
        auto hash2 = (*hasher)(state.get());
        REQUIRE(hash1 != hash2);
    }

    //! Check that the new API produces the same hash values as the old API
    template <typename T, typename HS>
    void test_new_api(DatasetDescription<typename T::Sim::Format> dimensions, HS & source, size_t num_tables) {
        auto vec = UnitVectorFormat::generate_random(dimensions.args);
        auto stored = to_stored_type<typename T::Sim::Format>(vec, dimensions);

        std::vector<uint32_t> expected;
        for (size_t rep = 0; rep < num_tables; rep++) {
            auto hasher = source.sample();
            auto state = source.reset(stored.get(), false);
            auto h = (*hasher)(state.get());
            expected.push_back(h);
        }
        std::vector<uint32_t> hashes;
        source.hash_repetitions(stored.get(), hashes);
        REQUIRE(hashes.size() == num_tables);

        for (size_t rep = 0; rep < num_tables; rep++) {
            auto hash1 = expected[rep];
            auto hash2 = hashes[rep];
            REQUIRE(hash1 == hash2);
        }
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
            auto state = source->reset(stored.get(), false);
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
            HashPoolArgs<SimHash>(50).build(dimensions, 1, 24));
        test_reset<FHTCrossPolytopeHash>(
            dimensions,
            HashPoolArgs<FHTCrossPolytopeHash>(80).build(dimensions, 1, 20));
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

    TEST_CASE("IndependentSource new api") {
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_description();

        size_t num_tables = 5;
        size_t num_bits = MAX_HASHBITS;

        IndependentHashSource<SimHash> simhash_source(
            dimensions,
            SimHashArgs(),
            num_tables,
            num_bits
        );
        test_new_api<SimHash, IndependentHashSource<SimHash>>(dimensions, simhash_source, num_tables);

        IndependentHashSource<FHTCrossPolytopeHash> cp_source(
            dimensions,
            FHTCrossPolytopeArgs(),
            num_tables,
            num_bits
        );
        test_new_api<FHTCrossPolytopeHash, IndependentHashSource<FHTCrossPolytopeHash>>(dimensions, cp_source, num_tables);
    }

    TEST_CASE("TensoredHashSource new api") {
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_description();

        size_t num_tables = 50;
        size_t num_bits = MAX_HASHBITS;

        TensoredHashSource<SimHash> simhash_source(
            dimensions,
            SimHashArgs(),
            num_tables,
            num_bits
        );
        test_new_api<SimHash, TensoredHashSource<SimHash>>(dimensions, simhash_source, num_tables);

        TensoredHashSource<FHTCrossPolytopeHash> cp_source(
            dimensions,
            FHTCrossPolytopeArgs(),
            num_tables,
            num_bits
        );
        test_new_api<FHTCrossPolytopeHash, TensoredHashSource<FHTCrossPolytopeHash>>(dimensions, cp_source, num_tables);
    }

    TEST_CASE("HashPool new api") {
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_description();

        size_t pool_size = 3000;
        size_t num_tables = 2;
        size_t num_bits = MAX_HASHBITS;

        HashPool<SimHash> simhash_source(
            dimensions,
            SimHashArgs(),
            pool_size,
            num_tables,
            num_bits
        );
        test_new_api<SimHash, HashPool<SimHash>>(dimensions, simhash_source, num_tables);

        HashPool<FHTCrossPolytopeHash> cp_source(
            dimensions,
            FHTCrossPolytopeArgs(),
            pool_size,
            num_tables,
            num_bits
        );
        test_new_api<FHTCrossPolytopeHash, HashPool<FHTCrossPolytopeHash>>(dimensions, cp_source, num_tables);
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
        unsigned int samples = 2900;
        Dataset<UnitVectorFormat> dataset(100);
        auto dimensions = dataset.get_description();
        test_hashes<SimHash>(
            dimensions,
            HashPoolArgs<SimHash>(3000).build(dimensions, samples, HASH_LENGTH),
            samples,
            HASH_LENGTH);
        test_hashes<FHTCrossPolytopeHash>(
            dimensions,
            HashPoolArgs<FHTCrossPolytopeHash>(3000).build(dimensions, samples, HASH_LENGTH),
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
