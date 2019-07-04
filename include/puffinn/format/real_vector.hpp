#pragma once

#include <vector>

#include "puffinn/format/generic.hpp"

namespace puffinn {
    struct RealVectorFormat {
        using Type = float;
        // 256 bit vectors
        const static unsigned int ALIGNMENT = 256/8;

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
            for (size_t i=0; i < dimensions.actual; i++) {
                storage[i] = input[i];
            }
            for (size_t i=dimensions.actual; i < dimensions.padded; i++) {
                storage[i] = 0.0;
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
