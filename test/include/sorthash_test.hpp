#pragma once

#include "catch.hpp"
#include "sorthash.hpp"
#include <random>
#include <algorithm>

namespace sorthash {
    TEST_CASE("Sort random hash values") {
        std::vector<uint32_t> numbers;
        size_t n = 100000;

        std::mt19937 generator (1234);
        std::uniform_int_distribution<uint32_t> distribution(0,1 << 23);

        for (size_t i = 0; i < n; i++) {
            numbers.push_back(distribution(generator));
        }
        std::vector<uint32_t> aux;

        puffinn::sort_hashes_24(numbers, aux);

        REQUIRE(aux.size() == n);
        // for (auto x : aux) {
        //     printf("%d\n", x);
        // }
        REQUIRE(std::is_sorted(aux.begin(), aux.end()));
    }

    TEST_CASE("Sort random hash value pairs") {
        std::vector<uint32_t> hashes;
        std::vector<uint32_t> indices;
        size_t n = 100000;

        std::mt19937 generator (1234);
        std::uniform_int_distribution<uint32_t> distribution(0,1 << 23);

        for (size_t i = 0; i < n; i++) {
            hashes.push_back(distribution(generator));
            indices.push_back(i);
        }
        std::vector<uint32_t> original_hashes(hashes);
        std::vector<uint32_t> hashes_out;
        std::vector<uint32_t> indices_out;

        puffinn::sort_hashes_pairs_24(hashes, hashes_out, indices, indices_out);

        REQUIRE(hashes_out.size() == n);
        REQUIRE(std::is_sorted(hashes_out.begin(), hashes_out.end()));
        // Check that hash values are still associated with the correct id
        for (size_t i = 0; i < n; i++) {
            REQUIRE(hashes_out[i] == original_hashes[indices_out[i]]);
        }
    }

}