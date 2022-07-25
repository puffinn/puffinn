#pragma once

#include <sstream>
#include "catch.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "puffinn/maxbuffercollection.hpp"

namespace Catch
{
    template<> struct StringMaker<puffinn::MaxBufferCollection::ResultPair>
    {
        static std::string convert(puffinn::MaxBufferCollection::ResultPair const& value)
        {
            std::stringstream sstream;
            sstream << "(" << value.first << ", " << value.second << ")";
            return sstream.str();
        }
    };

}

#define PRINTVEC(v) \
    for (size_t i=0; i < v.size(); i++) { \
        std::cout << #v << "[" << i << "] = " << v[i].first << " " << v[i].second << std::endl; \
    }


namespace puffinn {
    TEST_CASE("Single element, one buffer") {
        MaxBufferCollection buffer;
        buffer.init(1, 1);
        buffer.insert(0, 2, 0.6);
        auto best = buffer.best_entries(0);
        REQUIRE(best == std::vector<MaxBufferCollection::ResultPair>{{2, 0.6}});
    }

    TEST_CASE("Multiple elements, one buffer") {
        MaxBufferCollection buffer;
        buffer.init(1, 2);
        buffer.insert(0, 1, 0.06);
        buffer.insert(0, 3, 0.16);
        buffer.insert(0, 6, 0.56);
        buffer.insert(0, 2, 0.26);
        buffer.insert(0, 4, 0.36);
        buffer.insert(0, 5, 0.46);
        auto best = buffer.best_entries(0);
        REQUIRE(best == std::vector<MaxBufferCollection::ResultPair>{
            {6, 0.56},
            {5, 0.46}
        });
        REQUIRE(buffer.smallest_value(0) == 0.46f);
    }

    TEST_CASE("Multiple elements, two buffers") {
        MaxBufferCollection buffer;
        buffer.init(2, 2);
        buffer.insert(0, 1, 0.06);
        buffer.insert(0, 3, 0.16);
        buffer.insert(0, 6, 0.56);
        buffer.insert(0, 2, 0.26);
        buffer.insert(0, 4, 0.36);
        buffer.insert(0, 5, 0.46);
        buffer.insert(1, 1, 0.03);
        buffer.insert(1, 3, 0.13);
        buffer.insert(1, 6, 0.53);
        buffer.insert(1, 2, 0.23);
        buffer.insert(1, 4, 0.33);
        buffer.insert(1, 5, 0.43);
        REQUIRE(buffer.best_entries(0) == std::vector<MaxBufferCollection::ResultPair>{
            {6, 0.56},
            {5, 0.46}
        });
        REQUIRE(buffer.smallest_value(0) == 0.46f);
        REQUIRE(buffer.best_entries(1) == std::vector<MaxBufferCollection::ResultPair>{
            {6, 0.53},
            {5, 0.43}
        });
        REQUIRE(buffer.smallest_value(1) == 0.43f);
    }
    
    TEST_CASE("Multiple elements, two buffers, with duplicates") {
        MaxBufferCollection buffer;
        buffer.init(2, 2);
        buffer.insert(0, 1, 0.06);
        buffer.insert(0, 1, 0.06);
        buffer.insert(1, 2, 0.23);
        buffer.insert(0, 5, 0.46);
        buffer.insert(0, 1, 0.06);
        buffer.insert(0, 1, 0.06);
        buffer.insert(0, 2, 0.26);
        buffer.insert(0, 3, 0.16);
        buffer.insert(0, 4, 0.36);
        buffer.insert(1, 5, 0.43);
        buffer.insert(0, 5, 0.46);
        buffer.insert(0, 6, 0.56);
        buffer.insert(1, 1, 0.03);
        buffer.insert(1, 2, 0.23);
        buffer.insert(1, 6, 0.53);
        buffer.insert(0, 1, 0.06);
        buffer.insert(1, 3, 0.13);
        buffer.insert(1, 4, 0.33);
        buffer.insert(0, 1, 0.06);
        buffer.insert(1, 5, 0.43);
        buffer.insert(1, 5, 0.43);
        buffer.insert(1, 5, 0.43);
        buffer.insert(1, 6, 0.53);
        REQUIRE(buffer.best_entries(0) == std::vector<MaxBufferCollection::ResultPair>{
            {6, 0.56},
            {5, 0.46}
        });
        auto best1 = buffer.best_entries(1);
        // PRINTVEC(best1)
        REQUIRE(best1 == std::vector<MaxBufferCollection::ResultPair>{
            {6, 0.53},
            {5, 0.43}
        });
    }

}
