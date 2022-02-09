#pragma once
#include "catch.hpp"
#include "puffinn/pq_filter.hpp"
#include "puffinn/format/real_vector.hpp"
namespace pq{
    using namespace puffinn;

    TEST_CASE("See if codebook is stores or pointer losses reference") {
        unsigned int N = 8, dims = 8, m = 2, k  = 2;
        std::vector<float>  data[N] = {
                                        {-4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0},
                                        {-3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0},
                                        {-2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0},
                                        {-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0},
                                        {1.0 , 1.0, 2.0, 3.0, 4.0, 2.0, 7.0, 8.0},
                                        {2.0 , 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0},
                                        {3.0 , 1.0, 2.0, 3.0, 4.0, 1.0, 7.0, 8.0},
                                        {4.0 , 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0}};

        Dataset<RealVectorFormat> dataset(dims, N);
        for (auto entry: data){
            dataset.insert(entry);
        }
        PQFilter<RealVectorFormat> pq1(dataset, m,k);

    }
    
}
