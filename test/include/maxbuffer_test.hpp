#pragma once

#include "catch.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "puffinn/maxbuffer.hpp"

namespace maxbuffer {
    using namespace puffinn;

    TEST_CASE("Constructed with size 0") {
        MaxBuffer buffer(0);
        buffer.insert(1, 0.5);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{});
    }

    TEST_CASE("Empty MaxBuffer") {
        MaxBuffer buffer(2);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{});
    }

    TEST_CASE("Single element") {
        MaxBuffer buffer(2);
        buffer.insert(2, 0.6);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{{2, 0.6}});
        REQUIRE(buffer.smallest_value() == 0);
    }

    TEST_CASE("Retrieve while empty") {
        MaxBuffer buffer(2);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{});
        buffer.insert(100, 0.1);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{{100, 0.1}});
        REQUIRE(buffer.smallest_value() == 0.0);
    }

    TEST_CASE("Retrieve before filter") {
        MaxBuffer buffer(2);
        buffer.insert(100, 0.1);
        buffer.insert(50, 0.2);
        buffer.insert(105, 0.3);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{{105, 0.3}, {50, 0.2}});
        REQUIRE(buffer.smallest_value() == 0.2f);
    }

    TEST_CASE("Multiple filters") {
        MaxBuffer buffer(2);
        buffer.insert(1, 0.1);
        buffer.insert(2, 0.5);
        buffer.insert(3, 0.05);
        buffer.insert(4, 0.07);
        buffer.insert(5, 0.5);
        buffer.insert(6, 0.9);
        buffer.insert(7, 0.7);
        buffer.insert(8, 0.8);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{{6, 0.9}, {8, 0.8}});
        REQUIRE(buffer.smallest_value() == 0.8f);
    }

    TEST_CASE("Deduplication") {
        MaxBuffer buffer(2);
        buffer.insert(1, 0.1);
        buffer.insert(1, 0.1);
        buffer.insert(1, 0.1);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{{1, 0.1}});
    }

    // Mainly a test that it is well-behaved
    TEST_CASE("Out of range") {
        MaxBuffer buffer(2);
        buffer.insert(1, -5);
        buffer.insert(2, 1.2);
        REQUIRE(buffer.best_entries() == std::vector<MaxBuffer::ResultPair>{{2, 1.0}});
    }
}
