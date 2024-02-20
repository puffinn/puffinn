#pragma once

#include <ostream>

namespace puffinn {
    enum class HashSourceType {
        Independent,
        Pool,
        Tensor
    };

    template <typename T>
    struct HashSourceArgs;

    // A source for hash functions.
    //
    // This can be a useful to compute fewer hashes, at the cost of losing
    // independence between hashes.
    template <typename T>
    class HashSource {
    public:
        virtual ~HashSource() {}

        // Compute the LSH values for all tables supported by this pool for the given input vector.
        // Writes the results to the given output array, which can be reused
        virtual void hash_repetitions(
            const typename T::Sim::Format::Type * const input,
            std::vector<uint64_t> & output
        ) const = 0;

        virtual float collision_probability(
            float similarity,
            uint_fast8_t num_bits
        ) const = 0;

        // The probability that a point in the true top k was not found by looking at the given
        // number of tables.
        virtual float failure_probability(
            uint_fast8_t hash_length,
            uint_fast32_t tables,
            uint_fast32_t max_tables,
            // similarity to the k'th nearest neighbor found so far
            float kth_similarity
        ) const = 0;

        virtual uint_fast8_t get_bits_per_function() const = 0;

        // Probability of collision with a concatenated LSH function.
        float concatenated_collision_probability(uint_fast8_t num_bits, float similarity) const {
            auto bits_per_function = get_bits_per_function();
            auto whole_hashes = num_bits/bits_per_function;
            auto remaining_bits = num_bits%bits_per_function;

            float whole_hashes_prob = collision_probability(similarity, bits_per_function);
            float remaining_prob = collision_probability(similarity, remaining_bits);
            return std::pow(whole_hashes_prob, whole_hashes)*remaining_prob;
        }

        virtual void serialize(std::ostream&) const = 0;

    };

    /// Arguments that can be supplied with data from the ``LSHTable`` to construct a HashSource.
    /// @param T The used LSH family.
    template <typename T>
    struct HashSourceArgs {
        virtual ~HashSourceArgs() {}

        virtual std::unique_ptr<HashSource<T>> build(
            DatasetDescription<typename T::Sim::Format> desc,
            unsigned int num_tables,
            unsigned int num_bits
        ) const = 0;

        virtual std::unique_ptr<HashSourceArgs<T>> copy() const = 0;

        virtual uint64_t memory_usage(
            DatasetDescription<typename T::Sim::Format> dataset,
            unsigned int num_tables,
            unsigned int num_bits
        ) const = 0;

        virtual uint64_t function_memory_usage(
            DatasetDescription<typename T::Sim::Format> dataset,
            unsigned int num_bits
        ) const = 0;

        virtual void serialize(std::ostream& out) const = 0;

        virtual std::unique_ptr<HashSource<T>> deserialize_source(std::istream& in) const = 0;
    };
}
