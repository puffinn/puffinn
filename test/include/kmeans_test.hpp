#pragma once

#include "catch.hpp"
#include "puffinn/dataset.hpp"
#include "puffinn/kmeans.hpp"
#include "puffinn/format/real_vector.hpp"

#include <vector>
#include <iostream>
#include <set>

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

    TEST_CASE("basic kmeans clustering 2") {
        unsigned int N = 18, dims = 3, K=3;
        std::vector<float>  data[N] = { {2.0 , 2.0 , 0.0},
                                        {4.0 , 4.0 , 0.0},
                                        {2.0 , 4.0 , 0.0},
                                        {4.0 , 2.0 , 0.0},
                                        {3.0 , 3.0 , -1.0},
                                        {3.0 , 3.0 , 1.0},
                                        {2.0 , 0.0 , 2.0},
                                        {4.0 , 0.0 , 4.0},
                                        {2.0 , 0.0 , 4.0},
                                        {4.0 , 0.0 , 2.0},
                                        {3.0 , 1.0 , 3.0},
                                        {3.0 , -1.0 , 3.0},
                                        {0.0 , 2.0 , 2.0},
                                        {0.0 , 4.0 , 4.0},
                                        {0.0 , 2.0 , 4.0},
                                        {0.0 , 4.0 , 2.0},
                                        {1.0 , 3.0 , 3.0},
                                        {-1.0 , 3.0 , 3.0}};

        Dataset<RealVectorFormat> dataset(dims, N);
        for (auto entry : data) {
            dataset.insert(entry);
        }
        KMeans<RealVectorFormat> kmeans(dataset, (uint8_t)K);
        kmeans.fit();
        float   *cen1 = kmeans.getCentroid(0),
                *cen2 = kmeans.getCentroid(1),
                *cen3 = kmeans.getCentroid(2);
        std::vector<float> c1(cen1, cen1+3);
        std::vector<float> c2(cen2, cen2+3);
        std::vector<float> c3(cen3, cen3+3);
        std::set<std::vector<float>>
                                correct = {{0.0, 3.0, 3.0},
                                            {3.0, 0.0, 3.0},
                                            {3.0,3.0, 0.0}}, 
                                centroids = {c1,c2,c3};
            
        REQUIRE(correct == centroids);
    }
}