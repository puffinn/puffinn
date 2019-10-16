#pragma once

#include "catch.hpp"

#include "puffinn/dataset.hpp"
#include "puffinn/math.hpp"
#include "puffinn/format/unit_vector.hpp"
#include "puffinn/format/real_vector.hpp"

namespace math {
    using namespace puffinn;

    TEST_CASE("dot_product_i16 versions equal") {
        unsigned reps = 100;
        unsigned dims = 100;
        Dataset<UnitVectorFormat> dataset(dims);

        for (unsigned i=0; i < reps; i++) {
            auto a = UnitVectorFormat::generate_random(dims);
            auto b = UnitVectorFormat::generate_random(dims);
            auto sa = to_stored_type<UnitVectorFormat>(a, dataset.get_description());
            auto sb = to_stored_type<UnitVectorFormat>(b, dataset.get_description());

            int16_t simple = dot_product_i16_simple(sa.get(), sb.get(), dims);
            #ifdef __AVX2__
                int16_t avx2 = dot_product_i16_avx2(sa.get(), sb.get(), dims);
                REQUIRE(simple == avx2);
            #endif
        }
    }

    TEST_CASE("l2_distance_float versions equal") {
        unsigned reps = 100;
        unsigned dims = 100;
        Dataset<RealVectorFormat> dataset(dims);

        for (unsigned i=0; i < reps; i++) {
            auto a = RealVectorFormat::generate_random(dims);
            auto b = RealVectorFormat::generate_random(dims);
            auto sa = to_stored_type<RealVectorFormat>(a, dataset.get_description());
            auto sb = to_stored_type<RealVectorFormat>(b, dataset.get_description());

            float simple = l2_distance_float_simple(sa.get(), sb.get(), dims);
            #ifdef __AVX__
                float avx = l2_distance_float_avx(sa.get(), sb.get(), dims);
                // Order of operations differ, so small error is accetable.
                REQUIRE(simple == Approx(avx).epsilon(0.0001));
            #endif
        }
    }
}
