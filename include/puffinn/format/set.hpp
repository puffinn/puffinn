#pragma once

#include "puffinn/format/generic.hpp"

namespace puffinn {
    /// A format for storing sets.
    ///
    /// Currently, only ``std::vector<uint32_t>`` is supported as input type.
    /// Each integer in this set represents a token and must be
    /// between 0 and the number of dimensions specified when constructing the ``LSHTable``.
    struct SetFormat {
        // Stored in sorted order.
        using Type = std::vector<uint32_t>;
        const static unsigned int ALIGNMENT = 0;

        static unsigned int storage_dimensions(unsigned int) {
            return 1;
        }

        static void store(
            const std::vector<uint32_t>& set,
            std::vector<uint32_t>* storage,
            DatasetDimensions dimensions
        ) {
            for (auto v : set) {
                if (v >= dimensions.actual) {
                    throw std::invalid_argument("invalid token");
                }
            }
            // Placement-new
            auto vec = new(storage) std::vector<uint32_t>; 
            vec->reserve(set.size());
            for (auto i : set) {
                vec->push_back(i);
            }
            std::sort(vec->begin(), vec->end());
        }

        static std::vector<uint32_t> generate_random(unsigned int dimensions) {
            // Probability of each element to be included in the set.
            const float INCLUSION_PROB = 0.3;

            std::uniform_real_distribution<float> dist(0.0, 1.0);
            auto& rng = get_default_random_generator();

            std::vector<uint32_t> res;
            for (uint32_t i=0; i < dimensions; i++) {
                if (dist(rng) < INCLUSION_PROB) {
                    res.push_back(i);
                }
            }
            return res;
        }
    };
}
