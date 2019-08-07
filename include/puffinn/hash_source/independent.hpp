#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/hash_source/hash_source.hpp"

#include <memory>
#include <vector>

namespace puffinn {
    template <typename T>
    class IndependentHasher;

    // A source of completely independent hash functions.
    template <typename T>
    class IndependentHashSource : public HashSource<T> {
        T hash_family;
        std::vector<typename T::Function> hash_functions;
        unsigned int functions_per_hasher;
        uint_fast8_t bits_per_function;
        unsigned int next_function = 0;
        unsigned int bits_to_cut;

        typename T::Sim::Format::Type* hashed_vec = nullptr;
    
    public:
        IndependentHashSource(
            DatasetDescription<typename T::Sim::Format> desc,
            typename T::Args args,
            // Number of hashers to create.
            unsigned int num_hashers,
            // Number of bits per hasher.
            unsigned int num_bits
        ) 
          : hash_family(desc, args)
        {
            bits_per_function = hash_family.bits_per_function();
            functions_per_hasher =
                (num_bits+bits_per_function-1)/bits_per_function;
            auto num_functions = functions_per_hasher*num_hashers;
            bits_to_cut = bits_per_function*functions_per_hasher-num_bits;
            hash_functions.reserve(num_functions);
            for (unsigned int i=0; i < num_functions; i++) {
                hash_functions.push_back(hash_family.sample());
            }
        }

        uint64_t hash(unsigned int first_hash) const {
            uint64_t res = 0;
            for (unsigned int i=0; i < functions_per_hasher; i++) {
                res <<= bits_per_function;
                res |= hash_functions[first_hash+i](hashed_vec);
            }
            return (res >> bits_to_cut);
        }

        void reset(typename T::Sim::Format::Type* vec) {
            hashed_vec = vec;    
        }

        // Retrieve the number of functions this source can create.
        size_t get_size() {
            return hash_functions.size()/functions_per_hasher;
        }

        std::unique_ptr<Hash> sample() {
            auto res = std::make_unique<IndependentHasher<T>>(this, next_function);
            next_function += functions_per_hasher;
            return res;
        }

        uint_fast8_t get_bits_per_function() {
            return bits_per_function;
        }

        float collision_probability(
            float similarity,
            uint_fast8_t num_bits
        ) {
            return hash_family.collision_probability(similarity, num_bits);
        }

        float failure_probability(
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
            return false;
        }
    };

    template <typename T>
    class IndependentHasher : public Hash {
        IndependentHashSource<T>* source;
        unsigned int first_function;

    public:
        IndependentHasher(IndependentHashSource<T>* source, unsigned int first_function)
          : source(source),
            first_function(first_function)
        {
        }

        uint64_t operator()() const {
            return source->hash(first_function);
        }
    };

    /// Describes a hash source where all hash functions are sampled independently.
    template <typename T>
    struct IndependentHashArgs : public HashSourceArgs<T> {
        /// Arguments for the hash family.
        typename T::Args args;

        std::unique_ptr<HashSource<T>> build(
            DatasetDescription<typename T::Sim::Format> desc,
            unsigned int num_tables,
            unsigned int num_bits
        ) const {
            return std::make_unique<IndependentHashSource<T>> (
                desc,
                args,
                num_tables,
                num_bits
            );
        }

        std::unique_ptr<HashSourceArgs<T>> copy() const {
            return std::make_unique<IndependentHashArgs<T>>(*this);
        }
    };
}
