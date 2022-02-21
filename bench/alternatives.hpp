#pragma once
#include "puffinn/dataset.hpp"
#include "puffinn/hash_source/hash_source.hpp"

namespace puffinn {
    template <typename T>
    class IndependentHasherStatic {
        T hash_family;
        std::vector<typename T::Function> hash_functions;
        unsigned int functions_per_hasher;
        uint_fast8_t bits_per_function;
        unsigned int next_function = 0;
        unsigned int bits_to_cut;

    public:
        IndependentHasherStatic(
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

        uint64_t hash(
            unsigned int first_hash, 
            typename T::Sim::Format::Type * hashed_vec
        ) const {
            uint64_t res = 0;
            for (unsigned int i=0; i < functions_per_hasher; i++) {
                res <<= bits_per_function;
                res |= hash_functions[first_hash+i](hashed_vec);
            }
            return (res >> bits_to_cut);
        }
    };
}