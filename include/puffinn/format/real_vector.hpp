#pragma once

#include <vector>

#include "puffinn/format/generic.hpp"
#include "puffinn/math.hpp"

namespace puffinn {
    struct RealVectorFormat {
        using Type = float;
        using Args = unsigned int;
        // 256 bit vectors
        const static unsigned int ALIGNMENT = 256/8;

        static unsigned int storage_dimensions(Args dimensions) {
            return dimensions;
        }

        static void store(
            const std::vector<float>& input,
            Type* storage,
            DatasetDescription<RealVectorFormat> dataset
        ) {
            if (input.size() != dataset.args) {
                throw std::invalid_argument("input.size()");
            }
            for (size_t i=0; i < dataset.args; i++) {
                storage[i] = input[i];
            }
            for (size_t i=dataset.args; i < dataset.storage_len; i++) {
                storage[i] = 0.0;
            }
        }

        static void free(Type&) {}


        static float distance(const float* lhs, const float* rhs, unsigned int dimension)
        {
            return l2_distance_float_simple(lhs, rhs, dimension);
        }

        static void add_assign(float* const lhs, const float* rhs, unsigned int dimensions)
        {   
            add_assign_float_simple(lhs, rhs, dimensions);
        }
        static void subtract_assign(float * const lhs, const float* rhs, unsigned int dimensions){
            
            subtract_assign_float_simple(lhs,rhs, dimensions);

        }
        static void divide_assign(float* lhs, const unsigned int div, unsigned int dimensions)
        {
            multiply_assign_float_simple(lhs, 1.0/div, dimensions);
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
