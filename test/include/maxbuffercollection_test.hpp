#pragma once

#include "catch.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "puffinn/maxbuffercollection.hpp"

#define PRINTVEC(v) \
    for (size_t i=0; i < v.size(); i++) { \
        std::cout << #v << "[" << i << "] = " << v[i].first << " " << v[i].second << std::endl; \
    }


namespace puffinn {
    TEST_CASE("Single element, one buffer") {
        MaxBufferCollection buffer;
        buffer.init(1, 2);
        buffer.insert(0, 2, 0.6);
        auto best = buffer.best_entries(0);
        REQUIRE(best == std::vector<MaxBufferCollection::ResultPair>{{2, 0.6}});
    }

    TEST_CASE("Multiple elements, one buffer") {
        MaxBufferCollection buffer;
        buffer.init(1, 2);
        buffer.insert(0, 1, 0.6);
        buffer.insert(0, 3, 1.6);
        buffer.insert(0, 6, 5.6);
        buffer.insert(0, 2, 2.6);
        buffer.insert(0, 4, 3.6);
        buffer.insert(0, 5, 4.6);
        auto best = buffer.best_entries(0);
        REQUIRE(best == std::vector<MaxBufferCollection::ResultPair>{
            {6, 5.6},
            {5, 4.6}
        });
    }

    TEST_CASE("Multiple elements, two buffers") {
        MaxBufferCollection buffer;
        buffer.init(2, 2);
        buffer.insert(0, 1, 0.6);
        buffer.insert(0, 3, 1.6);
        buffer.insert(0, 6, 5.6);
        buffer.insert(0, 2, 2.6);
        buffer.insert(0, 4, 3.6);
        buffer.insert(0, 5, 4.6);
        buffer.insert(1, 1, 0.3);
        buffer.insert(1, 3, 1.3);
        buffer.insert(1, 6, 5.3);
        buffer.insert(1, 2, 2.3);
        buffer.insert(1, 4, 3.3);
        buffer.insert(1, 5, 4.3);
        REQUIRE(buffer.best_entries(0) == std::vector<MaxBufferCollection::ResultPair>{
            {6, 5.6},
            {5, 4.6}
        });
        REQUIRE(buffer.best_entries(1) == std::vector<MaxBufferCollection::ResultPair>{
            {6, 5.3},
            {5, 4.3}
        });
    }
    
    TEST_CASE("Multiple elements, two buffers, with duplicates") {
        MaxBufferCollection buffer;
        buffer.init(2, 2);
        buffer.insert(0, 1, 0.6);
        buffer.insert(0, 1, 0.6);
        buffer.insert(1, 2, 2.3);
        buffer.insert(0, 5, 4.6);
        buffer.insert(0, 1, 0.6);
        buffer.insert(0, 1, 0.6);
        buffer.insert(0, 2, 2.6);
        buffer.insert(0, 3, 1.6);
        buffer.insert(0, 4, 3.6);
        buffer.insert(1, 5, 4.3);
        buffer.insert(0, 5, 4.6);
        buffer.insert(0, 6, 5.6);
        buffer.insert(1, 1, 0.3);
        buffer.insert(1, 2, 2.3);
        buffer.insert(1, 6, 5.3);
        buffer.insert(0, 1, 0.6);
        buffer.insert(1, 3, 1.3);
        buffer.insert(1, 4, 3.3);
        buffer.insert(0, 1, 0.6);
        buffer.insert(1, 5, 4.3);
        buffer.insert(1, 5, 4.3);
        buffer.insert(1, 5, 4.3);
        buffer.insert(1, 6, 5.3);
        REQUIRE(buffer.best_entries(0) == std::vector<MaxBufferCollection::ResultPair>{
            {6, 5.6},
            {5, 4.6}
        });
        auto best1 = buffer.best_entries(1);
        // PRINTVEC(best1)
        REQUIRE(best1 == std::vector<MaxBufferCollection::ResultPair>{
            {6, 5.3},
            {5, 4.3}
        });
    }

}
