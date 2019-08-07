#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/hash_source/hash_source.hpp"

#include <memory>
#include <limits>
#include <random>

namespace puffinn {
    template<typename T>
    class PooledHasher;

    // A pool of hash functions that can be shared.
    // These functions can be mixed to produce different hashes, which means that fewer hash
    // computations are needed. However if the pool contains too few hash functions, it will
    // perform worse.
    template <typename T>
    class HashPool : public HashSource<T> {
        T hash_family;
        std::vector<typename T::Function> hash_functions;
        std::unique_ptr<LshDatatype> hashes;
        uint_fast8_t bits_per_function;
        unsigned int bits_per_hasher;

    public:
        HashPool(
            DatasetDescription<typename T::Sim::Format> desc,
            typename T::Args args,
            unsigned int num_functions,
            unsigned int bits_per_hasher
        )
          : hash_family(desc, args),
            hashes(std::unique_ptr<LshDatatype>(
                new LshDatatype[num_functions])),
            bits_per_function(hash_family.bits_per_function()),
            bits_per_hasher(bits_per_hasher)
        {
            num_functions /= hash_family.bits_per_function();
            hash_functions.reserve(num_functions);
            for (unsigned int i=0; i < num_functions; i++) {
                hash_functions.push_back(hash_family.sample());
            }
        }

        std::unique_ptr<Hash> sample() {
            return std::make_unique<PooledHasher<T>>(this, bits_per_hasher);
        }

        uint64_t concatenate_hash(const std::vector<unsigned int>& indices) {
            uint64_t res = 0;
            for (auto idx : indices) {
                res <<= bits_per_function;
                res |= hashes.get()[idx];
            }
            return res;
        }

        unsigned int get_size() {
            return hash_functions.size();
        }

        uint_fast8_t get_bits_per_function() {
            return bits_per_function;
        }

        // Recompute hashes for a new vector.
        void reset(typename T::Sim::Format::Type* vec) {
            #pragma omp parallel for
            for (size_t i=0; i<hash_functions.size(); i++) {
                hashes.get()[i] = hash_functions[i](vec);
            }
        }

        float collision_probability(
            float similarity,
            uint_fast8_t num_bits
        ) {
            return hash_family.collision_probability(similarity, num_bits);
        }

        // This assumes that hashes are independent, which is not true.
        // Therefore using a pool can result in recalls that are lower than expected.
        virtual float failure_probability(
            uint_fast8_t hash_length,
            uint_fast32_t tables,
            uint_fast32_t max_tables,
            float kth_similarity
        ) {
            float col_prob =
                this->concatenated_collision_probability(hash_length, kth_similarity);
            float last_prob =
                this->concatenated_collision_probability(hash_length+1, kth_similarity);
            return std::pow(1.0-col_prob, tables)*std::pow(1-last_prob, max_tables-tables);
        }

        bool precomputed_hashes() const {
            return true;
        }
    };

    // A hash function that selects some hashfunctions from a pool and reuses their values.
    template <typename T>
    class PooledHasher : public Hash {
        // The pool always outlives the hasher.
        HashPool<T> *pool;
        std::vector<unsigned int> indices;
        int bits_to_cut;

    public:
        // Create a hash function reusing the given pool.
        // The resulting hash consists of exactly the given number of bits.
        PooledHasher(HashPool<T> *pool, size_t hash_length)
          : pool(pool)
        {
            auto& rand_gen = get_default_random_generator();
            int bits_per_function = pool->get_bits_per_function();
            std::uniform_int_distribution<unsigned int> random_idx(0, pool->get_size()-1);
            for (size_t i=0; i < hash_length; i += bits_per_function) {
                indices.push_back(random_idx(rand_gen));
            }
            bits_to_cut = pool->get_bits_per_function()*indices.size()-hash_length;
        }

        // It is assumed that the vector to hash has already been hashed in the pool, so no
        // argument is needed.
        uint64_t operator()() const {
            return (pool->concatenate_hash(indices)) >> bits_to_cut;
        }
    };

    /// Describes a hash source which precomputes a pool of a given size.
    /// 
    /// Each hash is then constructed by sampling from this pool.
    /// This reduces the number of hashes that need to be computed, but produces hashes of lower quality.
    ///
    /// It is typically possible to choose a pool size which 
    /// performs better than independent hashing,
    /// but using independent hashes is a better default.
    template <typename T>
    struct HashPoolArgs : public HashSourceArgs<T> {
        /// Arguments for the hash family.
        typename T::Args args;
        /// The size of the pool in bits.
        unsigned int pool_size;

        constexpr HashPoolArgs(unsigned int pool_size)
          : pool_size(pool_size)
        {
        }

        std::unique_ptr<HashSource<T>> build(
            DatasetDescription<typename T::Sim::Format> desc,
            unsigned int /* num_tables */,
            unsigned int num_bits_per_function
        ) const {
            return std::make_unique<HashPool<T>> (
                desc,
                args,
                pool_size,
                num_bits_per_function
            );
        }

        std::unique_ptr<HashSourceArgs<T>> copy() const {
            return std::make_unique<HashPoolArgs<T>>(*this);
        }
    };
}
