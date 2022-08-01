#pragma once

#include "puffinn/format/set.hpp"
#include "puffinn/similarity_measure/jaccard.hpp"

#include <istream>
#include <ostream>
#include <random>

namespace puffinn {
    class MultiplyAddHash {
        uint64_t a;
        uint64_t b;

    public:
        MultiplyAddHash(std::mt19937_64& rng) {
            a = rng();
            b = rng();
        }

        MultiplyAddHash(std::istream& in) {
            in.read(reinterpret_cast<char*>(&a), sizeof(uint64_t));
            in.read(reinterpret_cast<char*>(&b), sizeof(uint64_t));
        }

        void serialize(std::ostream& out) const {
            out.write(reinterpret_cast<const char*>(&a), sizeof(uint64_t));
            out.write(reinterpret_cast<const char*>(&b), sizeof(uint64_t));
        }

        uint64_t operator()(uint32_t val) const {
            return (((uint64_t) val) * a + b) >> 32;
        }
    };

    class TabulationHash {
        uint64_t t1[256];
        uint64_t t2[256];
        uint64_t t3[256];
        uint64_t t4[256];

    public:
        TabulationHash(std::mt19937_64& rng) {
            for (size_t i=0; i < 256; i++) {
                t1[i] = rng();
                t2[i] = rng();
                t3[i] = rng();
                t4[i] = rng();
            }
        }

        TabulationHash(std::istream& in) {
            in.read(reinterpret_cast<char*>(&t1[0]), 256*sizeof(uint64_t));
            in.read(reinterpret_cast<char*>(&t2[0]), 256*sizeof(uint64_t));
            in.read(reinterpret_cast<char*>(&t3[0]), 256*sizeof(uint64_t));
            in.read(reinterpret_cast<char*>(&t4[0]), 256*sizeof(uint64_t));
        }

        void serialize(std::ostream& out) const {
            out.write(reinterpret_cast<const char*>(&t1[0]), 256*sizeof(uint64_t));
            out.write(reinterpret_cast<const char*>(&t2[0]), 256*sizeof(uint64_t));
            out.write(reinterpret_cast<const char*>(&t3[0]), 256*sizeof(uint64_t));
            out.write(reinterpret_cast<const char*>(&t4[0]), 256*sizeof(uint64_t));
        }

        uint64_t operator()(uint32_t val) const {
            return (
                t1[val & 0xFF] ^
                t2[(val >> 8) & 0xFF] ^
                t3[(val >> 16) & 0xFF] ^
                t4[(val >> 24) & 0xFF]);
        }
    };

    // Randomize the lower bits to another unique value.
    class BitPermutation {
        unsigned int num_bits;
        std::vector<uint32_t> perm;
    public:
        BitPermutation() {}

        BitPermutation(std::mt19937_64& rng, unsigned int universe_size, unsigned int num_bits)
          : num_bits(num_bits)
        {
            for (unsigned int i=0; i < std::min(universe_size, (1u << num_bits)); i++) {
                perm.push_back(i);
            }
            std::shuffle(perm.begin(), perm.end(), rng);
        }

        BitPermutation(std::istream& in) {
            in.read(reinterpret_cast<char*>(&num_bits), sizeof(num_bits));
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
            perm.reserve(len);
            for (size_t i=0; i < len; i++) {
                uint32_t v;
                in.read(reinterpret_cast<char*>(&v), sizeof(uint32_t));
                perm.push_back(v);
            }
        }

        void serialize(std::ostream& out) const {
            out.write(reinterpret_cast<const char*>(&num_bits), sizeof(unsigned int));
            size_t len = perm.size();
            out.write(reinterpret_cast<char*>(&len), sizeof(size_t));
            out.write(reinterpret_cast<const char*>(&perm[0]), len*sizeof(uint32_t));
        }

        LshDatatype operator()(LshDatatype v) const {
            if (num_bits != 0) {
                auto mask = (1 << num_bits)-1;
                auto lower = v & mask;
                auto lower_perm = perm[lower];
                // assemble permuted lower bits and the unchanged upper bits.
                return (v & (~mask)) | lower_perm;
            }
            return v;
        }
    };

    class MinHashFunction {
        MultiplyAddHash hash;
        BitPermutation permutation;

    public:
        MinHashFunction(MultiplyAddHash hash, BitPermutation perm) : hash(hash), permutation(perm) {
        }

        MinHashFunction(std::istream& in)
          : hash(in),
            permutation(in)
        {
        }

        void serialize(std::ostream& out) const {
            hash.serialize(out);
            permutation.serialize(out);
        }

        LshDatatype operator()(std::vector<uint32_t>* vec) const {
            uint64_t min_hash = 0xFFFFFFFFFFFFFFFF; // 2^64-1
            uint32_t min_token = 0;
            for (uint32_t i : *vec) {
                uint64_t h = hash(i);
                if (h < min_hash) {
                    min_hash = h;
                    min_token = i;
                }
            }
            return permutation(min_token);
        }
    };

    /// Arguments for ``MinHash``.
    struct MinHashArgs {
        /// Randomize a number of the lower bits in minhash values.
        ///
        /// For some pairs of sets the collision probability does not increase when using partial
        /// hashes.
        /// This means that the achieved recall can be lower than
        /// expected.
        /// By randomizing parts of the tokens, this becomes unlikely to happen.
        unsigned int randomized_bits;

        constexpr MinHashArgs()
          : randomized_bits(4)
        {
        }

        MinHashArgs(std::istream& in) {
            in.read(reinterpret_cast<char*>(&randomized_bits), sizeof(unsigned int));
        }

        void serialize(std::ostream& out) const {
            out.write(reinterpret_cast<const char*>(&randomized_bits), sizeof(unsigned int));
        }

        void set_no_preprocessing() {
            // randomize_tokens runs when sampling
        }

        uint64_t memory_usage(DatasetDescription<SetFormat> dataset) const {
            auto perm_len = std::min(dataset.args, (1u << randomized_bits));
            uint64_t perm_mem = perm_len * sizeof(uint32_t);
            return sizeof(MinHashFunction)+perm_mem;
        }
    };


    /// A multi-bit hash function for sets, which selects the minimum token 
    /// when ordered by a random permutation.
    /// 
    /// In practice, each token is hashed using a standard hash function,
    /// after which the token with the smallest hash is selected.
    /// The higher the Jaccard similarity, the higher the probability of two sets containing the same minimum token. 
    class MinHash {
    public:
        using Args = MinHashArgs;
        using Sim = JaccardSimilarity;
        using Function = MinHashFunction;

    private:
        Args args;
        unsigned int set_size;

    public:
        MinHash(DatasetDescription<SetFormat> dataset, Args args)
          : args(args),
            // Needs to hash to at least one bit, for which the
            // minimum set size is 2.
            set_size(std::max(dataset.args, 2u))
        {
        }

        MinHash(std::istream& in) {
            args = Args(in);
            in.read(reinterpret_cast<char*>(&set_size), sizeof(unsigned int));
        }

        void serialize(std::ostream& out) const {
            args.serialize(out);
            out.write(reinterpret_cast<const char*>(&set_size), sizeof(unsigned int));
        }

        Function sample() {
            std::mt19937_64 rng;
            rng.seed(get_default_random_generator()());

            BitPermutation perm(rng, set_size, args.randomized_bits);
            return Function(MultiplyAddHash(rng), perm);
        }

        unsigned int bits_per_function() const {
            return ceil_log(set_size);
        }

        float collision_probability(float similarity, int_fast8_t num_bits) const {
            // Number of hashes that would collide with the given number of bits.
            float num_possible_hashes =
                static_cast<float>(set_size)/std::min(1u << num_bits, set_size)-1.0;
            // Probability of selecting one of the colliding hashes.
            float miss_collision_prob = num_possible_hashes/(set_size-1);
            return similarity+(1-similarity)*miss_collision_prob;
        }

        float icollision_probability(float p) const {
            throw std::logic_error("not yet implemented");
        }
    };

    class MinHash1BitFunction {
        MinHashFunction hash;

    public: 
        MinHash1BitFunction(MinHashFunction hash)
          : hash(hash)
        {
        }

        MinHash1BitFunction(std::istream& in)
          : hash(in)
        {
        }

        void serialize(std::ostream& out) const {
            hash.serialize(out);
        }

        LshDatatype operator()(std::vector<uint32_t>* vec) const {
            return hash(vec)%2;
        }
    };

    /// ``MinHash``, but only use 1 bit to make it suitable for sketching. 
    class MinHash1Bit {
    public:
        using Args = MinHash::Args;
        using Sim = MinHash::Sim;
        using Function = MinHash1BitFunction;

    private:
        MinHash minhash;

    public:
        MinHash1Bit(DatasetDescription<SetFormat> dataset, Args args)
          : minhash(dataset, args)
        {
        }

        MinHash1Bit(std::istream& in)
          : minhash(in)
        {
        }

        void serialize(std::ostream& out) const {
            minhash.serialize(out);
        }

        Function sample() {
            return Function(minhash.sample());
        }

        unsigned int bits_per_function() const {
            return 1;
        }

        float collision_probability(float similarity, int_fast8_t num_bits) const {
            if (num_bits > 1) { num_bits = 1; }
            return minhash.collision_probability(similarity, num_bits);
        }

        float icollision_probability(float p) const {
            return 2.0 * p - 1.0;
        }
    };
}
