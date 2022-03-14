#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/hash_source/hash_source.hpp"
#include "puffinn/typedefs.hpp"
#include "puffinn/performance.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <istream>
#include <ostream>
#include <utility>
#include <vector>

namespace puffinn {
    // A query stores the hash, the current prefix as well as which segment in the map that has
    // already been searched.
    struct PrefixMapQuery {
        // The prefix of the query hash.
        LshDatatype hash;
        // Mask used to reduce hashes to the considered prefix.
        LshDatatype prefix_mask;
        // The index of the first and one past the last vector in the referenced hashes that share
        // the searched prefix.
        uint_fast32_t prefix_start;
        uint_fast32_t prefix_end;

        // Construct a query with the hashes precomputed.
        //
        // The main purpose is to avoid hashing multiple times.
        // A reference to the list of hashes is also stored to be able to find the next segment
        // in the map to process.
        PrefixMapQuery(
            LshDatatype hash,
            const std::vector<LshDatatype>& hashes,
            uint32_t prefix_index_start,
            uint32_t prefix_index_end
        )
          : hash(hash)
        {
            // given indices are just hints to where it lies between
            prefix_start = prefix_index_start;
            prefix_end = prefix_index_end;
            // inspired by databasearchitects.blogspot.com/2015/09/trying-to-speed-up-binary-search.html
            uint_fast32_t half = prefix_end-prefix_start;
            while (half != 0) {
                half /= 2;
                uint_fast32_t mid = prefix_start+half;
                prefix_start = (hashes[mid] < hash ? (mid+1) : prefix_start);
            }
            // Initially set to empty segment of index just above the prefix.
            prefix_end = prefix_start;
            prefix_mask = 0xffffffff;
        }
    };

    const static int SEGMENT_SIZE = 12;
    // A PrefixMap stores all inserted values in sorted order by their hash codes.
    //
    // This allows querying all values that share a common prefix. The length of the prefix
    // can be decreased to look at a larger set of values. When the prefix is decreased,
    // previously queried values are not queried again.
    template <typename T>
    class PrefixMap {
        using HashedVecIdx = std::pair<uint32_t, LshDatatype>;
        // Number of bits to precompute locations in the stored vector for.
        const static int PREFIX_INDEX_BITS = 13;

    public: // TODO private
        // contents
        std::vector<uint32_t> indices;
        std::vector<LshDatatype> hashes;
        // Scratch space for use when rebuilding. The length and capacity is set to 0 otherwise.
        std::vector<HashedVecIdx> rebuilding_data;

        // Length of the hash values used.
        unsigned int hash_length;
        std::unique_ptr<Hash> hash_function;

        // index of the first value with each prefix.
        // If there is no such value, it is the first higher prefix instead.
        // Used as a hint for the binary search.
        uint32_t prefix_index[(1 << PREFIX_INDEX_BITS)+1] = {0};

    public:
        // Construct a new prefix map over the specified dataset using the given hash functions.
        PrefixMap(std::unique_ptr<Hash> hash, unsigned int hash_length)
          : hash_length(hash_length),
            hash_function(std::move(hash))
        {
            // Ensure that the map can be queried even if nothing is inserted.
            rebuild();
        }

        PrefixMap(std::istream& in, HashSource<T>& source) {
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
            indices.resize(len);
            hashes.resize(len);
            if (len != 0) {
                in.read(reinterpret_cast<char*>(&indices[0]), len*sizeof(uint32_t));
                in.read(reinterpret_cast<char*>(&hashes[0]), len*sizeof(LshDatatype));
            }

            size_t rebuilding_len;
            in.read(reinterpret_cast<char*>(&rebuilding_len), sizeof(size_t));
            rebuilding_data.resize(rebuilding_len);
            if (rebuilding_len != 0) {
                in.read(
                    reinterpret_cast<char*>(&rebuilding_data[0]), 
                    rebuilding_len*sizeof(HashedVecIdx));
            }

            in.read(reinterpret_cast<char*>(&hash_length), sizeof(unsigned int));
            hash_function = source.deserialize_hash(in);

            in.read(
                reinterpret_cast<char*>(&prefix_index[0]),
                ((1 << PREFIX_INDEX_BITS)+1)*sizeof(uint32_t));
        }

        void serialize(std::ostream& out) const {
            size_t len = indices.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(size_t));
            if (len != 0) {
                out.write(reinterpret_cast<const char*>(&indices[0]), len*sizeof(uint32_t));
                out.write(reinterpret_cast<const char*>(&hashes[0]), len*sizeof(LshDatatype));
            }

            size_t rebuilding_len = rebuilding_data.size();
            out.write(reinterpret_cast<const char*>(&rebuilding_len), sizeof(size_t));
            if (rebuilding_len != 0) {
                out.write(
                    reinterpret_cast<const char*>(&rebuilding_data[0]),
                    rebuilding_len*sizeof(HashedVecIdx));
            }

            out.write(reinterpret_cast<const char*>(&hash_length), sizeof(unsigned int));
            hash_function->serialize(out);

            out.write(reinterpret_cast<const char*>(
                &prefix_index),
                ((1 << PREFIX_INDEX_BITS)+1)*sizeof(uint32_t));
        }

        // Add a vector to be included next time rebuild is called. 
        // Expects that the hash source was last reset with that vector.
        void insert(uint32_t idx, HashSourceState* hash_state) {
            rebuilding_data.push_back({ idx, (*hash_function)(hash_state) });
        }

        // Reserve the correct amount of memory before inserting.
        void reserve(size_t size) {
            if (hashes.size() == 0) {
                rebuilding_data.reserve(size);
            } else {
                rebuilding_data.reserve(size-(hashes.size()-2*SEGMENT_SIZE));
            }
        }

        void rebuild() {
            // A value whose prefix will never match that of a query vector, as long as less than 32
            // hash bits are used.
            static const LshDatatype IMPOSSIBLE_PREFIX = 0xffffffff;

            rebuilding_data.reserve(hashes.size()+rebuilding_data.size());
            if (hashes.size() != 0) {
                // Move data to temporary vector for sorting.
                for (size_t i=SEGMENT_SIZE; i < hashes.size()-SEGMENT_SIZE; i++) {
                    rebuilding_data.push_back({ indices[i], hashes[i] });
                }
            }
            
            std::sort(
                rebuilding_data.begin(),
                rebuilding_data.end(),
                [](HashedVecIdx& a, HashedVecIdx& b) {
                    return a.second < b.second;
                }
            );
            std::vector<LshDatatype> new_hashes;
            new_hashes.reserve(rebuilding_data.size()+2*SEGMENT_SIZE);
            std::vector<uint32_t> new_indices;
            new_indices.reserve(rebuilding_data.size()+2*SEGMENT_SIZE);

            // Pad with SEGMENT_SIZE values on each size to remove need for bounds check.
            for (int i=0; i < SEGMENT_SIZE; i++) {
                new_hashes.push_back(IMPOSSIBLE_PREFIX);
                new_indices.push_back(0);
            }
            for (auto v : rebuilding_data) {
                new_indices.push_back(v.first);
                new_hashes.push_back(v.second);
            }
            for (int i=0; i < SEGMENT_SIZE; i++) {
                new_hashes.push_back(IMPOSSIBLE_PREFIX);
                new_indices.push_back(0);
            }
            hashes = std::move(new_hashes);
            indices = std::move(new_indices);

            // Build prefix_index data structure.
            // Index of the first occurence of the prefix
            uint32_t idx = 0;
            for (unsigned int prefix=0; prefix < (1u << PREFIX_INDEX_BITS); prefix++) {
                while (
                    idx < rebuilding_data.size() &&
                    (hashes[SEGMENT_SIZE+idx] >> (hash_length-PREFIX_INDEX_BITS)) < prefix
                ) {
                    idx++;
                }
                prefix_index[prefix] = SEGMENT_SIZE+idx;
            }
            prefix_index[1 << PREFIX_INDEX_BITS] = SEGMENT_SIZE+rebuilding_data.size();

            rebuilding_data.clear();
            rebuilding_data.shrink_to_fit();
        }

        // Construct a query object to search for the nearest neighbors of the given vector.
        PrefixMapQuery create_query(HashSourceState* hash_state) const {
            g_performance_metrics.start_timer(Computation::Hashing);
            auto hash = (*hash_function)(hash_state);
            g_performance_metrics.store_time(Computation::Hashing);
            g_performance_metrics.start_timer(Computation::CreateQuery);
            auto prefix = hash >> (hash_length-PREFIX_INDEX_BITS);
            PrefixMapQuery res(
                hash,
                hashes,
                prefix_index[prefix],
                prefix_index[prefix+1]);
            g_performance_metrics.store_time(Computation::CreateQuery);
            return res;
        }

        // Reduce the length of the prefix by one and retrieve the range of indices that should
        // be considered next.
        // Assumes that everything in the current prefix is already searched. This is not true
        // in the first iteration, but will be after there has been a search each way.
        // As most queries need multiple iterations, this should not be a problem.
        std::pair<const uint32_t*, const uint32_t*> get_next_range(PrefixMapQuery& query) const {
            auto prev_mask = (query.prefix_mask >> 1);
            auto removed_bit = prev_mask & (-prev_mask); // Least significant bit
            // The value of the removed bit.
            // If a 0-bit is removed, search upwards, otherwise downwards.
            // (Since we need to include all values with 1-bits, which are above)
            // In the first iteration, where no bit is removed, this is 0.
            auto bit_value = query.hash & removed_bit;

            auto hash_prefix = (query.hash & query.prefix_mask);
            if (bit_value == 0) {
                auto next_idx = query.prefix_end;
                auto start_idx = next_idx;
                while ((hashes[next_idx] & query.prefix_mask) == hash_prefix) {
                    next_idx += SEGMENT_SIZE;
                }
                auto end_idx = next_idx;
                if (end_idx >= indices.size()-SEGMENT_SIZE) {
                    // Adjust the range so that no values in the padding are checked
                    // However, next time the padding is reached it would cause end_idx < start_idx
                    end_idx = std::max(start_idx, end_idx-SEGMENT_SIZE);
                }
                query.prefix_mask <<= 1;
                return std::make_pair(&indices[start_idx], &indices[end_idx]);
            } else {
                auto next_idx = query.prefix_start-1;
                auto end_idx = next_idx+1;
                while ((hashes[next_idx] & query.prefix_mask) == hash_prefix) {
                    next_idx -= SEGMENT_SIZE;
                }
                auto start_idx = next_idx+1;
                if (start_idx < SEGMENT_SIZE) {
                    start_idx = std::min(end_idx, start_idx+SEGMENT_SIZE);
                }
                query.prefix_mask <<= 1;
                return std::make_pair(&indices[start_idx], &indices[end_idx]);
            }
        }

        std::pair<const uint32_t*, const uint32_t*> get_segment(size_t left, size_t right) {
            return std::make_pair(&indices[left], &indices[right]);
        }

        static uint64_t memory_usage(size_t size, uint64_t function_size) {
            size = size+2*SEGMENT_SIZE;
            return sizeof(PrefixMap)
                + size*sizeof(uint32_t)
                + size*sizeof(LshDatatype)
                + function_size; 
        }
    };
}
