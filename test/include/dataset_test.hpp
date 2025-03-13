#pragma once

#include "catch.hpp"

#include "puffinn/dataset.hpp"
#include "puffinn/format/unit_vector.hpp"

#include <cstring>
#include <random>

namespace dataset {
    using namespace puffinn;

    TEST_CASE("Dataset accessor") {
        const unsigned int DIMENSIONS = 3;
        const unsigned int CAPACITY = 1000;

        Dataset<UnitVectorFormat> dataset(DIMENSIONS, CAPACITY);
        REQUIRE(dataset.get_description().args == 3);
        REQUIRE(dataset.get_description().storage_len == 16);
        REQUIRE(dataset.get_size() == 0);
        REQUIRE(dataset.get_capacity() == CAPACITY);
        std::vector<std::vector<float>> vectors = {
            { 1., 0., 0. },
            { 0., 0., -1. },
            { 0., 1., 0. }
        };
        for (auto vec : vectors) { dataset.insert(vec); }
        REQUIRE(dataset[0][0] == UnitVectorFormat::to_16bit_fixed_point(1.0));
        REQUIRE(dataset[1][2] == UnitVectorFormat::to_16bit_fixed_point(-1.0));
        REQUIRE(dataset[2][1] == UnitVectorFormat::to_16bit_fixed_point(1.0));
    }

    TEST_CASE("Dynamic resizing") {
        std::mt19937_64 rng;
        const unsigned int DIMENSIONS = 120;
        const unsigned int SIZE = 10000;

        Dataset<UnitVectorFormat> dataset(DIMENSIONS);

        std::vector<float> vec(DIMENSIONS, 0);
        vec[1] = 1.0;
        dataset.insert(vec);
        for (unsigned int i=0; i < SIZE; i++) {
            dataset.insert(UnitVectorFormat::generate_random(DIMENSIONS, rng));
        }

        REQUIRE(dataset.get_size() == SIZE+1);
        REQUIRE(dataset.get_capacity() >= SIZE+1);
        // Initial vector still there.
        REQUIRE(dataset[0][1] == UnitVectorFormat::to_16bit_fixed_point(1.0));
    }
}
