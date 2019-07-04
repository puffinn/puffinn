#pragma once

#include <cassert>
#include <cmath>
#include <random>
#include <vector>

#include "puffinn/format/generic.hpp"
#include "puffinn/typedefs.hpp"

namespace puffinn {
    // Bytes in a 256-bit vector
    const int VECTOR256_ALIGNMENT = 256/8;

    /// A format for storing real vectors of unit length.
    /// 
    /// Currently, only ``std::vector<float>``` is supported as input type. 
    /// The vectors do not need to be normalized before insertion.
    ///
    /// Each number is stored using a 16-bit fixed point format. 
    /// Although this slightly reduces the precision,
    /// the inaccuracies are very unlikely to have an impact on the result.
    /// The vectors are stored using 256-bit alignment.
    struct UnitVectorFormat {
        // Represent the values as signed 15bit fixed point numbers between -1 and 1.
        // Done since the values are always in the range [-1, 1].
        // This is equivalent to what is used by `mulhrs`. However this cannot represent 1 exactly.
        using Type = int16_t;

        const static unsigned int ALIGNMENT = VECTOR256_ALIGNMENT;

        // Number of `Type` values that fit into a 256 bit vector.
        const static unsigned int VALUES_PER_VEC = 16;

        // Convert a floating point value between -1 and 1 to the internal, fixed point representation.
        static constexpr int16_t to_16bit_fixed_point(float val) {
            assert(val >= -1.0 && val <= 1.0);
            // The value cannot represent 1, so for accuracy, positive numbers are compared to the value
            // above INT16_MAX. 1.0 is set to the greatest representable value.
            auto res = (int16_t)(-val*INT16_MIN);
            if (val > 0.0 && res < 0) { res = INT16_MAX; }
            return res;
        }

        // Convert a number between -1 and 1 from the internal,
        // fixed point representation to floating point.
        static constexpr float from_16bit_fixed_point(Type val) {
            return -((float)val)/INT16_MIN;
        }

        static unsigned int storage_dimensions(unsigned int dimensions) {
            return dimensions;
        }

        static void store(
            const std::vector<float>& input,
            Type* storage,
            DatasetDimensions dimensions
        ) {
            if (input.size() != dimensions.actual) {
                throw std::invalid_argument("input.size()");
            }

            std::vector<float> copy = input;
            float len_squared = 0.0;
            for (auto v : copy) {
                len_squared += v*v;
            }

            auto len = std::sqrt(len_squared);
            if (len != 0.0) {
                for (auto& v : copy) {
                    v /= len;
                }
            }

            for (size_t i=0; i < copy.size(); i++) {
                storage[i] = to_16bit_fixed_point(copy[i]);
            }
            for (size_t i=copy.size(); i < dimensions.padded; i++) {
                storage[i] = to_16bit_fixed_point(0.0);
            }
        }

        static std::vector<float> generate_random(unsigned int dimensions) {
            std::normal_distribution<float> normal_distribution(0.0, 1.0);
            auto& generator = get_default_random_generator();

            std::vector<float> values;
            for (unsigned int i=0; i<dimensions; i++) {
                values.push_back(normal_distribution(generator));
            }
            return values;
        }
    };
}
