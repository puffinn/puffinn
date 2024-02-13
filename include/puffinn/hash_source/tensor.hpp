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
        unsigned int num_hashers;
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
            num_hashers(num_hashers),
            num_bits(num_bits)
        {
            for (unsigned int i=0; i < independent_hash_source.get_size(); i++) {
                hashers.push_back(independent_hash_source.sample());
            }
        }

        TensoredHashSource(std::istream& in)
          : independent_hash_source(in)
        {
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
            hashers.reserve(len);
            for (size_t i=0; i < len; i++) {
                // these functions only use the args for dispatch, so
                // it does not matter that it is not the 'correct' arguments.
                hashers.push_back(independent_hash_source.deserialize_hash(in));
            }
            in.read(reinterpret_cast<char*>(&num_hashers), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&next_hash_idx), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&num_bits), sizeof(unsigned int));
        }

        void serialize(std::ostream& out) const {
            independent_hash_source.serialize(out);
            size_t len = hashers.size();
            out.write(reinterpret_cast<char*>(&len), sizeof(size_t));
            for (auto& h : hashers) {
                h->serialize(out);
            }
            out.write(reinterpret_cast<const char*>(&num_hashers), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&next_hash_idx), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&num_bits), sizeof(unsigned int));
        }

        std::unique_ptr<Hash> sample() {
            auto index_pair = get_minimal_index_pair(next_hash_idx);
            next_hash_idx++;
            return std::make_unique<TensoredHasher<T>>(
                this,
                index_pair.first,
                index_pair.second);
        }

        void hash_repetitions(
            const typename T::Sim::Format::Type * const input,
            std::vector<uint64_t> & output
        ) const {
            // In order to avoid allocating a new vector to hold the tensored data
            // every time we hash something, we make the output vector a little bit larger:
            // enough to store both the output **and** the tensored repetitions.
            // After computing the tensored repetitions directly in the output
            // vector, we move them to the end, and place the actual output hashes
            // in the first part of `output`. Finally, we trim the size so that
            // the caller does not see the scratch work. Note that calling `resize`
            // does not de-allocate the memory, so on the next call we will not make
            // an allocation again.
            size_t tensored_hashers = independent_hash_source.get_size();
            independent_hash_source.hash_repetitions(input, output);
            output.resize(num_hashers + tensored_hashers);
            for (size_t i=0; i<tensored_hashers; i++) {
                output[num_hashers+i] = intersperse_zero(output[i]);
            }
            size_t right_start = tensored_hashers / 2;

            if (num_bits % 2 == 0) {
                for (size_t i=0; i < tensored_hashers / 2; i++) {
                    output[num_hashers + i] <<= 1;
                }
            } else {
                for (size_t i=right_start; i < tensored_hashers; i++) {
                    output[num_hashers + i] >>= 1;
                }
            }

            for(size_t rep=0; rep < num_hashers; rep++) {
                auto index_pair = get_minimal_index_pair(rep);
                uint32_t h_left = output[num_hashers + index_pair.first];
                uint32_t h_right = output[num_hashers + right_start + index_pair.second];
                uint32_t h = h_left | h_right;
                output[rep] = h;
            }
            output.resize(num_hashers);
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

        std::unique_ptr<Hash> deserialize_hash(std::istream& in) const {
            return std::make_unique<TensoredHasher<T>>(in, this);
        }
    };

    template <typename T>
    class TensoredHasher : public Hash {
        const TensoredHashSource<T>* source;
        unsigned int lhs_idx;
        unsigned int rhs_idx;

    public:
        TensoredHasher(TensoredHashSource<T>* source, unsigned int lhs_idx, unsigned int rhs_idx)
          : source(source),
            lhs_idx(lhs_idx),
            rhs_idx(rhs_idx)
        {
        }

        TensoredHasher(std::istream& in, const TensoredHashSource<T>* source)
          : source(source)
        {
            in.read(reinterpret_cast<char*>(&lhs_idx), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&rhs_idx), sizeof(unsigned int));
        }

        void serialize(std::ostream& out) const {
            out.write(reinterpret_cast<const char*>(&lhs_idx), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&rhs_idx), sizeof(unsigned int));
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

        TensoredHashArgs() = default;

        TensoredHashArgs(std::istream& in)
          : args(in)
        {
        }

        void serialize(std::ostream& out) const {
            HashSourceType type = HashSourceType::Tensor;
            out.write(reinterpret_cast<char*>(&type), sizeof(HashSourceType));
            args.serialize(out);
        }

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

        std::unique_ptr<HashSource<T>> deserialize_source(std::istream& in) const {
            return std::make_unique<TensoredHashSource<T>>(in);
        }
    };
}
