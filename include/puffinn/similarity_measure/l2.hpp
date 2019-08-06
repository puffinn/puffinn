#pragma once

#include "puffinn/format/real_vector.hpp"
#include "puffinn/hash/crosspolytope.hpp"
#include "puffinn/hash/simhash.hpp"
#include "puffinn/math.hpp"

#include <cmath>

namespace puffinn {
    struct L2Distance {
        using Format = RealVectorFormat;
        using DefaultHash = FHTCrossPolytopeHash;
        using DefaultSketch = SimHash;

        static float compute_similarity(float* lhs, float* rhs, DatasetDescription<Format> desc) {
            auto dist = l2_distance_float_sse(lhs, rhs, desc.args);
            // Convert to a similarity between 0 and 1,
            // which is needed to calculate collision probabilities.
            return 1.0/(dist+1.0);
        }
    };
}

