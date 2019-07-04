#pragma once

#include "catch.hpp"

#include "puffinn/format/generic.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "puffinn/similarity_measure/l2.hpp"
#include "puffinn/similarity_measure/jaccard.hpp"

namespace similarity_measure {
    using namespace puffinn;

    TEST_CASE("CosineSimilarity::compute_similarity") {
        std::vector<float> v1{0.2, 0.4, 0.0, 0.8, 0.4};
        std::vector<float> v2{0.4, 0.0, 0.8, 0.2, 0.4};
        Dataset<UnitVectorFormat> dataset(5);
        auto stored_1 = to_stored_type<UnitVectorFormat>(v1, dataset.get_dimensions());
        auto stored_2 = to_stored_type<UnitVectorFormat>(v2, dataset.get_dimensions());
        float res16 = CosineSimilarity::compute_similarity(
            stored_1.get(), stored_2.get(), v1.size());
        float expected = 0.7;
        REQUIRE(std::abs(res16-expected) <= 1e-4);

        // Test larger array
        std::vector<float> long_1(64, 0);
        long_1[2] = 0.2;
        long_1[16] = 0.4;
        long_1[27] = 0.8;
        long_1[63] = 0.4;
        std::vector<float> long_2(64, 0);
        long_2[2] = 0.4;
        long_2[27] = 0.2;
        long_2[51] = 0.8;
        long_2[63] = 0.4;

        dataset = Dataset<UnitVectorFormat>(64);
        stored_1 = to_stored_type<UnitVectorFormat>(long_1, dataset.get_dimensions());
        stored_2 = to_stored_type<UnitVectorFormat>(long_2, dataset.get_dimensions());
        float res64 = CosineSimilarity::compute_similarity(
            stored_1.get(), stored_2.get(), long_1.size());
        REQUIRE(std::abs(res64-expected) <= 1e-4);
    }

    TEST_CASE("L2Distance::compute_similarity") {
        auto v1 = allocate_storage<RealVectorFormat>(1, 32);
        auto v2 = allocate_storage<RealVectorFormat>(1, 32);
        for (unsigned int i=0; i<32; i++) {
            v2.get()[i] = v1.get()[i] = 0;
        }

        v1.get()[0] = 0.2;
        v2.get()[0] = 0.5;
        v1.get()[21] = -1.5;
        v2.get()[21] = 0.4;

        float res16 = L2Distance::compute_similarity(v1.get(), v2.get(), 16);
        float res32 = L2Distance::compute_similarity(v1.get(), v2.get(), 32);
        REQUIRE(std::abs(res16-(1.0/1.09)) <= 1e-6);
        REQUIRE(std::abs(res32-(1.0/4.70)) <= 1e-6);
    }

    TEST_CASE("JaccardSimilarity::compute_similarity") {
        Dataset<SetFormat> dataset(100);
        auto d = dataset.get_dimensions();

        std::vector<uint32_t> a, b;
        SetFormat::store({}, &a, d);
        SetFormat::store({}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == 0);

        SetFormat::store({1, 2, 3}, &a, d);
        SetFormat::store({1, 2, 3}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == 1.0);

        SetFormat::store({1, 2, 3}, &a, d);
        SetFormat::store({4, 5, 3}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == Approx(1.0/5.0));

        SetFormat::store({1}, &a, d);
        SetFormat::store({1, 2, 3, 4, 5, 6}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == Approx(1.0/6.0));

        SetFormat::store({1, 2, 3}, &a, d);
        SetFormat::store({4, 5, 6}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == 0.0);

        SetFormat::store({5, 7, 1}, &a, d);
        SetFormat::store({1, 2, 3, 4, 5, 6}, &b, d);
        REQUIRE(JaccardSimilarity::compute_similarity(&a, &b, 1) == Approx(2.0/7.0));
    }
}


