#pragma once

#include <vector>
#include <istream>
#include <ostream>

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
        /// Size of the universe.
        using Args = unsigned int;
        const static unsigned int ALIGNMENT = 0;

        static unsigned int storage_dimensions(Args) {
            return 1;
        }

        static uint64_t inner_memory_usage(Type& vec) {
            return vec.capacity()*sizeof(uint32_t);
        }

        static void store(
            const std::vector<uint32_t>& set,
            std::vector<uint32_t>* storage,
            DatasetDescription<SetFormat> dataset
        ) {
            for (auto v : set) {
                if (v >= dataset.args) {
                    throw std::invalid_argument("invalid token");
                }
            }
            // Placement-new
            free(*storage);
            auto vec = new(storage) std::vector<uint32_t>; 
            vec->reserve(set.size());
            for (auto i : set) {
                vec->push_back(i);
            }
            std::sort(vec->begin(), vec->end());
        }

        static void free(Type& vec) {
            vec.~vector();
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

        static void serialize_args(std::ostream& out, const Args& args) {
            out.write(reinterpret_cast<const char*>(&args), sizeof(Args));
        }

        static void deserialize_args(std::istream& in, Args* args) {
            in.read(reinterpret_cast<char*>(args), sizeof(Args));
        }

        static void serialize_type(std::ostream& out, const Type& type) {
            size_t len = type.size();
            out.write(reinterpret_cast<char*>(&len), sizeof(size_t));
            for (size_t i=0; i < len; i++) {
                out.write(reinterpret_cast<const char*>(&type[i]), sizeof(uint32_t));
            }
        }

        static void deserialize_type(std::istream& in, Type* type) {
            auto vec = new(type) std::vector<uint32_t>;
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
            vec->reserve(len);
            for (size_t i=0; i < len; i++) {
                uint32_t v;
                in.read(reinterpret_cast<char*>(&v), sizeof(uint32_t));
                vec->push_back(v);
            }
        }
    };

    template <>
    std::vector<uint32_t> convert_stored_type<SetFormat, std::vector<uint32_t>>(
        typename SetFormat::Type* storage,
        DatasetDescription<SetFormat>
    ) {
        return *storage;
    }
}
