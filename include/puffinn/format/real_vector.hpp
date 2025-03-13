#pragma once

#include <random>
#include <vector>

#include "puffinn/typedefs.hpp"
#include "puffinn/format/generic.hpp"

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

        static std::vector<float> generate_random(unsigned int dimensions, std::mt19937_64 &rng) {
            std::normal_distribution<float> normal_distribution(0.0, 1.0);
            std::vector<float> values;
            for (unsigned int i=0; i<dimensions; i++) {
                values.push_back(normal_distribution(rng));
            }
            return values;
        }
    };
}
