#pragma once

#include "catch.hpp"
#include "puffinn/hash/simhash.hpp"
#include "puffinn/hash/crosspolytope.hpp"
#include "puffinn/hash/minhash.hpp"
#include "puffinn/hash_source/pool.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "puffinn/similarity_measure/jaccard.hpp"

#include <cstdlib>
#include <random>

using namespace puffinn;

namespace hash {
    template <typename T>
    void test_hash_even_distribution(unsigned int dimensions) {
        const float ACCEPTED_DEVIATION = 0.03;
        std::mt19937_64 rng;

        Dataset<typename T::Sim::Format> dataset(dimensions);

        typename T::Args args;
        auto family = T(dataset.get_description(), args);
        typename T::Function hasher = family.sample(rng);

        auto hash_bits = family.bits_per_function();
        REQUIRE(hash_bits > 0);
        unsigned int possible_hashes = (1 << hash_bits);

        auto samples_per_hash = possible_hashes == 2 ? 5000 : 200;
        auto num_samples = possible_hashes*samples_per_hash;
        std::vector<int> hash_counts(possible_hashes);

        // Fill in random distribution
        std::vector<int> uniform_distribution(possible_hashes);
        std::uniform_int_distribution<int> distribution(0, possible_hashes-1);
        for (unsigned int i=0; i < num_samples; i++) {
            uniform_distribution[distribution(rng)]++;
        }
        std::sort(uniform_distribution.begin(), uniform_distribution.end());

        // Compute distribution for hashes
        for (unsigned int i=0; i < num_samples; i++) {
            auto vec = T::Sim::Format::generate_random(dimensions, rng);
            auto stored_vec = to_stored_type<typename T::Sim::Format>(vec, dataset.get_description());
            auto hash = hasher(stored_vec.get());
            REQUIRE(hash < possible_hashes);
            hash_counts[hash]++;
        }

        // Measure difference of distributions by summing the absolute difference.
        std::sort(hash_counts.begin(), hash_counts.end());

        unsigned int diff = 0;
        for (unsigned int pos_hash=0; pos_hash < possible_hashes; pos_hash++) {
            auto count = hash_counts[pos_hash];
            diff += std::abs(uniform_distribution[pos_hash]-count);
        }
        REQUIRE(diff <= ACCEPTED_DEVIATION*num_samples);
    }

    template <typename T, typename TSim>
    void test_hash_collision_probability(
        unsigned int dimensions,
        unsigned int num_samples = 10000,
        unsigned int num_bits = 0,
        typename T::Args args = typename T::Args()
    ) {
        const float ACCEPTED_DEVIATION = 0.02;
        std::mt19937_64 rng;

        Dataset<typename T::Sim::Format> dataset(dimensions);

        auto family = T(dataset.get_description(), args);
        auto hash_bits = num_bits == 0 ? family.bits_per_function() : num_bits;

        float prob_sum = 0;
        float actual_sum = 0;
        for (unsigned int i=0; i < num_samples; i++) {
            auto hasher = family.sample(rng);
            auto vec_a = T::Sim::Format::generate_random(dimensions, rng);
            auto vec_b = T::Sim::Format::generate_random(dimensions, rng);

            auto stored_a = to_stored_type<typename T::Sim::Format>(
                vec_a, dataset.get_description());
            auto stored_b = to_stored_type<typename T::Sim::Format>(
                vec_b, dataset.get_description());
            auto hash_a = hasher(stored_a.get()) % (1 << hash_bits);
            auto hash_b = hasher(stored_b.get()) % (1 << hash_bits);
            auto sim = TSim::compute_similarity(
                stored_a.get(), stored_b.get(), dataset.get_description());
            float prob = family.collision_probability(sim, hash_bits);
            prob_sum += prob;
            if (hash_a == hash_b) {
                actual_sum++;
            }
        }
        REQUIRE(std::abs(prob_sum-actual_sum) <= ACCEPTED_DEVIATION*num_samples);
    }

    TEST_CASE("Simhash evenly distributed") {
        test_hash_even_distribution<SimHash>(100);
    }

    TEST_CASE("Simhash collision probability") {
        test_hash_collision_probability<SimHash, CosineSimilarity>(100);
        test_hash_collision_probability<SimHash, CosineSimilarity>(3);
    }

    TEST_CASE("CrossPolytope evenly distributed") {
        test_hash_even_distribution<CrossPolytopeHash>(100);
    }

    TEST_CASE("CrossPolytope collision probability") {
        test_hash_collision_probability<CrossPolytopeHash, CosineSimilarity>(100);
    }

    TEST_CASE("FHTCrossPolytope evenly distributed") {
        test_hash_even_distribution<FHTCrossPolytopeHash>(16);
    }

    TEST_CASE("FHTCrossPolytope collision probability") {
        test_hash_collision_probability<FHTCrossPolytopeHash, CosineSimilarity>(100);
    }

    TEST_CASE("MinHash collision probability") {
        Dataset<SetFormat> dataset(100);
        MinHash minhash(dataset.get_description(), MinHashArgs());
        REQUIRE(minhash.collision_probability(0.0, 7) == 0.0);
        REQUIRE(minhash.collision_probability(0.5, 7) == 0.5);
        REQUIRE(minhash.collision_probability(1.0, 7) == 1.0);
        REQUIRE(minhash.collision_probability(0.5, 1)-(0.5+0.5*(49.0/99.0)) < 1e-6);

        MinHashArgs args;
        test_hash_collision_probability<MinHash, JaccardSimilarity>(100, 4000, 7, args);
        test_hash_collision_probability<MinHash, JaccardSimilarity>(100, 4000, 3, args);
        test_hash_collision_probability<MinHash1Bit, JaccardSimilarity>(100, 4000, 1, args);
    }

    TEST_CASE("bits_per_function") {
        unsigned int dimensions = 100;
        Dataset<UnitVectorFormat> dataset(dimensions);

        FHTCrossPolytopeHash fht_hash(dataset.get_description(), FHTCrossPolytopeArgs());
        REQUIRE(fht_hash.bits_per_function() == 8);

        CrossPolytopeHash cp_hash(dataset.get_description(), CrossPolytopeArgs());
        REQUIRE(cp_hash.bits_per_function() == 8);

        SimHash simhash = SimHash(dataset.get_description(), SimHashArgs());
        REQUIRE(simhash.bits_per_function() == 1);

        Dataset<SetFormat> set_dataset(dimensions);
        MinHash minhash = MinHash(set_dataset.get_description(), MinHashArgs());
        REQUIRE(minhash.bits_per_function() == 7);

        MinHash1Bit minhash1bit = MinHash1Bit(set_dataset.get_description(), MinHashArgs());
        REQUIRE(minhash1bit.bits_per_function() == 1);
    }

    TEST_CASE("Corner case") {
        unsigned int dimensions = 100;
        Dataset<SetFormat> dataset(dimensions);

        std::mt19937_64 rng;

        MinHashArgs args;
        auto family = MinHash1Bit(dataset.get_description(), args);

        auto a = to_stored_type<SetFormat>(std::vector<uint32_t>{0}, dataset.get_description());
        auto b = to_stored_type<SetFormat>(
            std::vector<uint32_t>{0, 1, 3, 5, 7}, 
            dataset.get_description());
        auto sim = JaccardSimilarity::compute_similarity(a.get(), b.get(), dataset.get_description());

        int samples = 1000;
        float collisions = 0;
        for (int i=0; i < samples; i++) {
            auto h = family.sample(rng);
            if (h(a.get()) == h(b.get())) { collisions++; }
        }
        float expected = samples*family.collision_probability(sim, 1);
        REQUIRE(std::abs(collisions-expected) < 0.05*samples);
    }
}
