#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/hash_source/hash_source.hpp"

#include <memory>
#include <limits>
#include <random>

namespace puffinn {
    template<typename T>
    class PooledHasher;

    struct HashPoolState : HashSourceState {
        std::unique_ptr<LshDatatype> hashes;
    };

    // A pool of hash functions that can be shared.
    // These functions can be mixed to produce different hashes, which means that fewer hash
    // computations are needed. However if the pool contains too few hash functions, it will
    // perform worse.
    template <typename T>
    class HashPool : public HashSource<T> {
        T hash_family;
        std::vector<typename T::Function> hash_functions;
        std::vector<std::vector<unsigned int>> indices;
        unsigned int num_tables;
        uint_fast8_t bits_per_function;
        unsigned int bits_per_hasher;
        unsigned int current_sampling_rep = 0;
        unsigned int bits_to_cut;

    public:
        HashPool(
            DatasetDescription<typename T::Sim::Format> desc,
            typename T::Args args,
            unsigned int num_functions,
            unsigned int num_tables,
            unsigned int bits_per_hasher
        )
          : hash_family(desc, args),
            num_tables(num_tables),
            bits_per_function(hash_family.bits_per_function()),
            bits_per_hasher(bits_per_hasher)
        {
            num_functions /= hash_family.bits_per_function();
            hash_functions.reserve(num_functions);
            for (unsigned int i=0; i < num_functions; i++) {
                hash_functions.push_back(hash_family.sample());
            }

            auto& rand_gen = get_default_random_generator();
            std::uniform_int_distribution<unsigned int> random_idx(0, num_functions-1);
            
            indices.reserve(num_tables);
            for (size_t rep = 0; rep < num_tables; rep++) {
                std::vector<unsigned int> rep_indices;
                rep_indices.reserve(bits_per_hasher);
                for (size_t i=0; i < bits_per_hasher; i += bits_per_function) {
                    rep_indices.push_back(random_idx(rand_gen));
                }
                indices.push_back(rep_indices);
            }

            bits_to_cut = bits_per_function * indices[0].size() - bits_per_hasher;

            // TODO: remove this is needed just to reset the state when testing, so
            // that if we call sample many times, we get the same sequences of indices we just constructed 
            reset_random_generator();
        }

        HashPool(std::istream& in)
          : hash_family(in)
        {
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
            hash_functions.reserve(len);
            for (size_t i=0; i < len; i++) {
                hash_functions.emplace_back(in);
            }
            size_t len_indices;
            in.read(reinterpret_cast<char*>(&len_indices), sizeof(size_t));
            for (size_t i=0; i < len_indices; i++) {
                std::vector<unsigned int> rep_indices;
                size_t len_index_array;
                in.read(reinterpret_cast<char*>(&len_index_array), sizeof(size_t));
                for (size_t j=0; j < len_index_array; j++) {
                    unsigned int idx;
                    in.read(reinterpret_cast<char*>(&idx), sizeof(unsigned int));
                    rep_indices.push_back(idx);
                }
                indices.push_back(rep_indices);
            }

            in.read(reinterpret_cast<char*>(&num_tables), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&bits_per_function), sizeof(uint_fast8_t));
            in.read(reinterpret_cast<char*>(&bits_per_hasher), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&current_sampling_rep), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&bits_to_cut), sizeof(unsigned int));
        }

        void serialize(std::ostream& out) const {
            hash_family.serialize(out);
            size_t len = hash_functions.size();
            out.write(reinterpret_cast<char*>(&len), sizeof(size_t));
            for (auto& h : hash_functions) {
                h.serialize(out);
            }
            size_t len_indices = indices.size();
            out.write(reinterpret_cast<char*>(&len_indices), sizeof(size_t));
            for (auto& index_vec : indices) {
                size_t len_index_vec = index_vec.size();
                out.write(reinterpret_cast<char*>(&len_index_vec), sizeof(size_t));
                for (auto i : index_vec) {
                    out.write(reinterpret_cast<const char*>(&i), sizeof(unsigned int));
                }
            }
            out.write(reinterpret_cast<const char*>(&num_tables), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&bits_per_function), sizeof(uint_fast8_t));
            out.write(reinterpret_cast<const char*>(&bits_per_hasher), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&current_sampling_rep), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&bits_to_cut), sizeof(unsigned int));
        }

        std::unique_ptr<Hash> sample() {
            return std::make_unique<PooledHasher<T>>(this, indices[current_sampling_rep++]);
        }

        uint64_t concatenate_hash(
            const std::vector<unsigned int>& indices,
            const LshDatatype* hashes
        ) const {
            uint64_t res = 0;
            for (auto idx : indices) {
                res <<= bits_per_function;
                res |= hashes[idx];
            }
            return res;
        }

        unsigned int get_size() const {
            return hash_functions.size();
        }

        uint_fast8_t get_bits_per_function() const {
            return bits_per_function;
        }

        uint_fast8_t get_bits_per_hasher() const {
            return bits_per_hasher;
        }

        void hash_repetitions(
            // TODO: Make this argument const. Currently it does not compile if we make it 
            // const because the hash function accepts a mutable pointer
            typename T::Sim::Format::Type * input,
            std::vector<LshDatatype> & output
        ) const {
            output.clear();

            // TODO: remove this allocation and reuse the scratch space
            std::vector<LshDatatype> pool;
            pool.reserve(hash_functions.size());

            for (size_t i = 0; i < hash_functions.size(); i++) {
                pool.push_back(hash_functions[i](input));
            }

            for (size_t rep = 0; rep < num_tables; rep++) {
                // Concatenate the hashes
                uint32_t res = 0;
                for (auto idx : indices[rep]) {
                    res <<= bits_per_function;
                    res |= pool[idx];
                }
                output.push_back(res >> bits_to_cut);
            }
        }

        // Recompute hashes for a new vector.
        std::unique_ptr<HashSourceState> reset(
                typename T::Sim::Format::Type* vec,
                bool parallelize
        ) const {
            auto hashes = std::unique_ptr<LshDatatype>(new LshDatatype[hash_functions.size()]);
                
            if (parallelize) {
                #pragma omp parallel for
                for (size_t i=0; i<hash_functions.size(); i++) {
                    hashes.get()[i] = hash_functions[i](vec);
                }
            } else {
                for (size_t i=0; i<hash_functions.size(); i++) {
                    hashes.get()[i] = hash_functions[i](vec);
                }
            }

            auto state = std::make_unique<HashPoolState>();
            state->hashes = std::move(hashes);
            return state;
        }

        float icollision_probability(float p) const {
            return hash_family.icollision_probability(p);
        }

        float collision_probability(
            float similarity,
            uint_fast8_t num_bits
        ) const {
            return hash_family.collision_probability(similarity, num_bits);
        }

        // This assumes that hashes are independent, which is not true.
        // Therefore using a pool can result in recalls that are lower than expected.
        virtual float failure_probability(
            uint_fast8_t hash_length,
            uint_fast32_t tables,
            uint_fast32_t max_tables,
            float kth_similarity
        ) const {
            float col_prob =
                this->concatenated_collision_probability(hash_length, kth_similarity);
            float last_prob =
                this->concatenated_collision_probability(hash_length+1, kth_similarity);
            return std::pow(1.0-col_prob, tables)*std::pow(1-last_prob, max_tables-tables);
        }

        bool precomputed_hashes() const {
            return true;
        }

        std::unique_ptr<Hash> deserialize_hash(std::istream& in) const {
            return std::make_unique<PooledHasher<T>>(in, this);
        }
    };

    // A hash function that selects some hashfunctions from a pool and reuses their values.
    template <typename T>
    class PooledHasher : public Hash {
        // The pool always outlives the hasher.
        const HashPool<T> *pool;
        std::vector<unsigned int> indices;
        int bits_to_cut;

    public:
        // Create a hash function reusing the given pool.
        // The resulting hash consists of exactly the given number of bits.
        PooledHasher(const HashPool<T> *pool, std::vector<unsigned int> & indices)
          : pool(pool), indices(indices)
        {
            bits_to_cut = pool->get_bits_per_function()*indices.size()-pool->get_bits_per_hasher();
        }

        PooledHasher(std::istream& in, const HashPool<T>* pool)
          : pool(pool)
        {
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
            indices.resize(len);
            in.read(reinterpret_cast<char*>(&indices[0]), len*sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&bits_to_cut), sizeof(int));

        }

        void serialize(std::ostream& out) const {
            size_t len = indices.size();
            out.write(reinterpret_cast<char*>(&len), sizeof(size_t));
            out.write(reinterpret_cast<const char*>(&indices[0]), len*sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&bits_to_cut), sizeof(int));
        }

        // It is assumed that the vector to hash has already been hashed in the pool, so no
        // argument is needed.
        uint64_t operator()(HashSourceState* state) const {
            auto pool_state = static_cast<HashPoolState*>(state);
            return (pool->concatenate_hash(indices, pool_state->hashes.get())) >> bits_to_cut;
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

        HashPoolArgs(std::istream& in)
          : args(in)
        {
            in.read(reinterpret_cast<char*>(&pool_size), sizeof(unsigned int));
        }

        void serialize(std::ostream& out) const {
            HashSourceType type = HashSourceType::Pool;
            out.write(reinterpret_cast<const char*>(&type), sizeof(HashSourceType));

            args.serialize(out);
            out.write(reinterpret_cast<const char*>(&pool_size), sizeof(unsigned int));
        }

        std::unique_ptr<HashSource<T>> build(
            DatasetDescription<typename T::Sim::Format> desc,
            unsigned int num_tables,
            unsigned int num_bits_per_function
        ) const {
            return std::make_unique<HashPool<T>> (
                desc,
                args,
                pool_size,
                num_tables,
                num_bits_per_function
            );
        }

        std::unique_ptr<HashSourceArgs<T>> copy() const {
            return std::make_unique<HashPoolArgs<T>>(*this);
        }

        uint64_t memory_usage(
            DatasetDescription<typename T::Sim::Format> dataset,
            unsigned int /*num_tables*/,
            unsigned int /*num_bits*/
        ) const {
            typename T::Args args_copy(args);
            args_copy.set_no_preprocessing();
            auto bits = T(dataset, args_copy).bits_per_function();
            return sizeof(HashPool<T>)
                + pool_size/bits*args.memory_usage(dataset);
        }

        uint64_t function_memory_usage(
            DatasetDescription<typename T::Sim::Format> dataset,
            unsigned int num_bits
        ) const {
            typename T::Args args_copy(args);
            args_copy.set_no_preprocessing();
            auto bits = T(dataset, args_copy).bits_per_function();
            return sizeof(PooledHasher<T>)+(num_bits+bits-1)/bits*sizeof(unsigned int);
        }

        std::unique_ptr<HashSource<T>> deserialize_source(std::istream& in) const {
            return std::make_unique<HashPool<T>>(in);
        }
    };
}
