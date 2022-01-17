#pragma once

#include "catch.hpp"
#include "puffinn/dataset.hpp"
#include "puffinn/kmeans.hpp"
#include "puffinn/format/real_vector.hpp"

#include <vector>
#include <iostream>


using namespace puffinn;
namespace kmeans {

    TEST_CASE("basic kmeans clustering") {
        unsigned int N = 8, dims = 2, K=2;
        std::vector<float>  data[N] = { {-4.0 , 1.0 },
                                        {-3.0 , 1.0 },
                                        {-2.0 , 1.0 },
                                        {-1.0 , 1.0 },
                                        {1.0 , 1.0 },
                                        {2.0 , 1.0 },
                                        {3.0 , 1.0 },
                                        {4.0 , 1.0 }};

        Dataset<RealVectorFormat> dataset(dims, N);
        for (auto entry : data) {
            dataset.insert(entry);
        }
        KMeans<RealVectorFormat> kmeans(dataset, (uint8_t)K);
        kmeans.fit();
        float   cen1 = kmeans.getCentroid(0)[0],
                cen2 = kmeans.getCentroid(1)[0];
        bool is_correct = (cen1 == 2.5 && cen2 == -2.5) || (cen1 == -2.5 && cen2 == 2.5);
        REQUIRE(is_correct);
    }

}