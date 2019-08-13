#pragma once

#include "puffinn/hash_source/independent.hpp"

namespace puffinn {
    uint64_t intersperse_zero(int64_t val) {
        uint64_t mask = 1;
        uint64_t shift = 0;

        uint64_t res = 0;
        for (unsigned i=0; i < sizeof(uint64_t)*8/2; i++) {
            res |= (val & mask) << shift;
            mask <<= 1;
            shift++;
        }
        return res;
    }

    // Helper function for getting indices to tensor.
    //
    // Retrieve the pair of indices where both sides are incremented as little as possible.
    // The rhs index is incremented first
    // Eg. (0, 0) (0, 1) (1, 0) (1, 1) (0, 2)
    static std::pair<unsigned int, unsigned int> get_minimal_index_pair(int idx) {
        int sqrt = static_cast<int>(std::sqrt(idx));
        if (idx == sqrt*sqrt+2*sqrt) {
            return {sqrt, sqrt};
        } else if (idx >= sqrt*sqrt+sqrt) {
            return {sqrt, idx-(sqrt*sqrt+sqrt)};
        } else { // idx >= sqrt*sqrt, always true
            return {idx-sqrt*sqrt, sqrt};
        }
    }

    template <typename T>
    class TensoredHasher;

    struct TensoredHashState : HashSourceState {
        std::vector<uint64_t> hashes;
    };

    // Contains two sets of hashfunctions. Hash values are constructed by interleaving one hash
    // from the first set with one from the second set. The used hash values are chosen so as
    // to avoid using the same combination twice.
    template <typename T>
    class TensoredHashSource : public HashSource<T> {
        IndependentHashSource<T> independent_hash_source;
        std::vector<std::unique_ptr<Hash>> hashers;
        unsigned int next_hash_idx = 0;
        unsigned int num_bits;

    public:
        TensoredHashSource(
            DatasetDescription<typename T::Sim::Format> dimensions,
            typename T::Args args,
            // Number of hashers to create.
            unsigned int num_hashers,
            // Number of bits per hasher.
            unsigned int num_bits
        ) 
          : independent_hash_source(
                dimensions,
                args,
                2*std::ceil(std::sqrt(static_cast<float>(num_hashers))),
                (num_bits+1)/2),
            num_bits(num_bits)
        {
            for (unsigned int i=0; i < independent_hash_source.get_size(); i++) {
                hashers.push_back(independent_hash_source.sample());
            }
        }

        std::unique_ptr<Hash> sample() {
            auto index_pair = get_minimal_index_pair(next_hash_idx);
            next_hash_idx++;
            return std::make_unique<TensoredHasher<T>>(
                this,
                index_pair.first,
                index_pair.second);
        }

        std::unique_ptr<HashSourceState> reset(typename T::Sim::Format::Type* vec) const {
            auto inner_state = independent_hash_source.reset(vec);

            std::vector<uint64_t> hashes;
            hashes.resize(hashers.size());
            // Store the hashes so that the final hash can be created by simply bitwise or-ing them together.
            #pragma omp parallel for
            for (unsigned int i=0; i < hashers.size(); i++) {
                hashes[i] = intersperse_zero((*hashers[i])(inner_state.get()));
            }
            // Store hashes shifted by one, so that lhs hashes and rhs hashes
            // do not overlap.
            // Ensure that the lhs hashes are longer or equal to the length of the rhs hashes.
            if (num_bits%2 == 0) {
                // Shift lhs hashes
                for (unsigned int i=0; i < hashers.size()/2; i++) {
                    hashes[i] <<= 1;
                }
            } else {
                // Shift rhs hashes the other way to reduce the size as we rounded up before.
                for (unsigned int i=hashers.size()/2; i < hashers.size(); i++) {
                    hashes[i] >>= 1;
                }
            }

            auto state = std::make_unique<TensoredHashState>();
            state->hashes = std::move(hashes);
            return state;
        }

        uint64_t hash(unsigned int lhs_idx, unsigned int rhs_idx, TensoredHashState* state) const {
            return state->hashes[lhs_idx] | state->hashes[state->hashes.size()/2+rhs_idx];
        }

        float collision_probability(
            float similarity,
            uint_fast8_t num_bits
        ) const {
            return independent_hash_source.collision_probability(similarity, num_bits);
        }

        float failure_probability(
            uint_fast8_t hash_length,
            uint_fast32_t num_tables, 
            uint_fast32_t max_tables,
            float similarity
        ) const {
            auto cur_left_bits = (hash_length+1)/2;
            auto cur_right_bits = hash_length-cur_left_bits;

            auto last_left_bits = (hash_length+2)/2;
            auto last_right_bits = hash_length+1-last_left_bits;

            auto cur_hashes = std::floor(std::sqrt(num_tables));
            auto last_hashes = std::floor(std::sqrt(max_tables))-cur_hashes;

            auto left_prob =
                this->concatenated_collision_probability(cur_left_bits, similarity);
            auto left_last_prob =
                this->concatenated_collision_probability(last_left_bits, similarity);

            auto right_prob =
                this->concatenated_collision_probability(cur_right_bits, similarity);
            auto right_last_prob =
                this->concatenated_collision_probability(last_right_bits, similarity);

            auto cur_upper_left_prob = 1.0-std::pow(1.0-left_prob, cur_hashes);
            auto last_upper_left_prob = 1.0-std::pow(1.0-left_last_prob, cur_hashes);
            auto last_lower_left_prob= 1.0-std::pow(1.0-left_last_prob, last_hashes);
            auto cur_upper_right_prob= 1.0-std::pow(1.0-right_prob, cur_hashes);
            auto last_upper_right_prob= 1.0-std::pow(1.0-right_last_prob, cur_hashes);
            auto last_lower_right_prob= 1.0-std::pow(1.0-right_last_prob, last_hashes);
            return
                (1-cur_upper_left_prob*cur_upper_right_prob) *
                (1-last_upper_left_prob*last_upper_right_prob) *
                (1-last_lower_left_prob*last_upper_right_prob) *
                (1-last_lower_left_prob*last_lower_right_prob);
        }

        uint_fast8_t get_bits_per_function() const {
            return independent_hash_source.get_bits_per_function();
        }

        bool precomputed_hashes() const {
            return true;
        }
    };

    template <typename T>
    class TensoredHasher : public Hash {
        TensoredHashSource<T>* source;
        unsigned int lhs_idx;
        unsigned int rhs_idx;

    public:
        TensoredHasher(TensoredHashSource<T>* source, unsigned int lhs_idx, unsigned int rhs_idx)
          : source(source),
            lhs_idx(lhs_idx),
            rhs_idx(rhs_idx)
        {
        }

        uint64_t operator()(HashSourceState* state) const {
            auto tensored_state = static_cast<TensoredHashState*>(state);
            return source->hash(lhs_idx, rhs_idx, tensored_state);
        }
    };

    /// Describes a hash source where hashes are constructed by combining a unique pair of smaller hashes from two sets.
    ///
    /// This means that the number of necessary hashes is only the square root of the number used for independent hashing. 
    /// However, this hash source does not perform well when targeting a high recall.
    template <typename T>
    struct TensoredHashArgs : public HashSourceArgs<T> {
        typename T::Args args;

        std::unique_ptr<HashSource<T>> build(
            DatasetDescription<typename T::Sim::Format> desc,
            unsigned int num_tables,
            unsigned int num_bits
        ) const {
            return std::make_unique<TensoredHashSource<T>> (
                desc,
                args,
                num_tables,
                num_bits
            );
        }

        std::unique_ptr<HashSourceArgs<T>> copy() const {
            return std::make_unique<TensoredHashArgs<T>>(*this);
        }

        uint64_t memory_usage(
            DatasetDescription<typename T::Sim::Format> dataset,
            unsigned int num_tables,
            unsigned int num_bits
        ) const {
            IndependentHashArgs<T> inner_args;
            auto inner_size = 2*std::ceil(std::sqrt(static_cast<float>(num_tables)));
            return sizeof(TensoredHashSource<T>)
                + inner_args.memory_usage(dataset, inner_size, (num_bits+1)/2)
                + inner_size*inner_args.function_memory_usage(dataset, num_bits);
        }

        uint64_t function_memory_usage(
            DatasetDescription<typename T::Sim::Format>,
            unsigned int /*num_bits*/
        ) const {
            return sizeof(TensoredHasher<T>);
        }
    };
}
