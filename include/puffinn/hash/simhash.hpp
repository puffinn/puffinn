#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/format/unit_vector.hpp"
#include "puffinn/math.hpp"
#include "puffinn/similarity_measure/cosine.hpp"

namespace puffinn {
    class SimHashFunction {
        std::unique_ptr<typename UnitVectorFormat::Type, decltype(free)*> hash_vec;
        unsigned int dimensions;

    public:
        SimHashFunction(DatasetDescription<UnitVectorFormat> dataset)
          : hash_vec(allocate_storage<UnitVectorFormat>(1, dataset.storage_len)),
            dimensions(dataset.args)
        {
            auto vec = UnitVectorFormat::generate_random(dimensions);
            UnitVectorFormat::store(vec, hash_vec.get(), dataset);
        }

        // Hash the given vector.
        LshDatatype operator()(int16_t* vec) const {
            auto dot = dot_product_i16(hash_vec.get(), vec, dimensions);
            return dot >= UnitVectorFormat::to_16bit_fixed_point(0.0);
        }
    };

    /// ``SimHash`` does not take any arguments.
    struct SimHashArgs {
        uint64_t memory_usage(DatasetDescription<UnitVectorFormat> dataset) const {
            return sizeof(SimHashFunction) + dataset.storage_len*sizeof(UnitVectorFormat::Type);
        }

        void set_no_preprocessing() {}
    };

    /// A one-bit hash function, which creates a random hyperplane at the origin
    /// and hashes points depending on which side of the plane the point is located on.
    class SimHash {
    public:
        using Args = SimHashArgs;
        using Sim = CosineSimilarity;
        using Function = SimHashFunction;

    private:
        DatasetDescription<UnitVectorFormat> dataset;

    public:
        SimHash(DatasetDescription<UnitVectorFormat> dataset, Args)
          : dataset(dataset)
        {
        }

        SimHashFunction sample() {
            return SimHashFunction(dataset);
        }

        unsigned int bits_per_function() {
            return 1;
        }

        float collision_probability(float similarity, int_fast8_t num_bits) const {
            if (num_bits == 0) {
                return 1.0;
            } else {
                return 1.0-std::acos(2*similarity-1)/M_PI;
            }
        }
    };
}
