#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/hash_source/hash_source.hpp"
#include "puffinn/typedefs.hpp"
#include "puffinn/performance.hpp"
#include "puffinn/sorthash.hpp"

#include "omp.h"
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
        // std::vector<HashedVecIdx> rebuilding_data;
        std::vector<std::vector<HashedVecIdx>> parallel_rebuilding_data;

        // Length of the hash values used.
        unsigned int hash_length;

        // index of the first value with each prefix.
        // If there is no such value, it is the first higher prefix instead.
        // Used as a hint for the binary search.
        uint32_t prefix_index[(1 << PREFIX_INDEX_BITS)+1] = {0};

    public:
        // Construct a new prefix map over the specified dataset using the given hash functions.
        PrefixMap(unsigned int hash_length)
          : hash_length(hash_length)
        {
            // Ensure that the map can be queried even if nothing is inserted.
            rebuild();
            auto max_threads = omp_get_max_threads();
            parallel_rebuilding_data.resize(max_threads);
        }

        PrefixMap(std::istream& in) {
            size_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
            indices.resize(len);
            hashes.resize(len);
            if (len != 0) {
                in.read(reinterpret_cast<char*>(&indices[0]), len*sizeof(uint32_t));
                in.read(reinterpret_cast<char*>(&hashes[0]), len*sizeof(LshDatatype));
            }

            // TODO Handle serialization
            size_t rebuilding_len;
            in.read(reinterpret_cast<char*>(&rebuilding_len), sizeof(size_t));
            // rebuilding_data.resize(rebuilding_len);
            if (rebuilding_len != 0) {
                for (size_t i=0; i<rebuilding_len; i++) {
                    HashedVecIdx v;
                    in.read(reinterpret_cast<char*>(&v), sizeof(HashedVecIdx));
                    parallel_rebuilding_data[0].push_back(v);
                }
            }

            in.read(reinterpret_cast<char*>(&hash_length), sizeof(unsigned int));

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

            size_t rebuilding_len = 0;
            for (auto & rd : parallel_rebuilding_data) {
                rebuilding_len += rd.size();
            }
            out.write(reinterpret_cast<const char*>(&rebuilding_len), sizeof(size_t));
            if (rebuilding_len != 0) {
                for (auto & rd : parallel_rebuilding_data) {
                    for (size_t i = 0; i <rd.size(); i++) {
                        out.write(reinterpret_cast<const char*>(&rd[i]), sizeof(HashedVecIdx));
                    }
                }
            }

            out.write(reinterpret_cast<const char*>(&hash_length), sizeof(unsigned int));

            out.write(reinterpret_cast<const char*>(
                &prefix_index),
                ((1 << PREFIX_INDEX_BITS)+1)*sizeof(uint32_t));
        }

        // Add a hash value, and associated index, to be included next time rebuild is called. 
        void insert(int tid, uint32_t idx, LshDatatype hash_value) {
            parallel_rebuilding_data[tid].push_back({ idx, hash_value });
        }

        // Reserve the correct amount of memory before inserting.
        void reserve(size_t size) {
            // TODO Divide equally across the vectors
            for (auto & rd : parallel_rebuilding_data) {
                rd.reserve(size);
            }
        }

        void rebuild() {
            // A value whose prefix will never match that of a query vector, as long as less than 32
            // hash bits are used.
            static const LshDatatype IMPOSSIBLE_PREFIX = 0xffffffff;

            size_t rebuilding_data_size = 0;
            for (auto & rd : parallel_rebuilding_data) {
                rebuilding_data_size += rd.size();
            }

            std::vector<LshDatatype> tmp_hashes;
            std::vector<uint32_t> tmp_indices;
            std::vector<LshDatatype> out_hashes;
            std::vector<uint32_t> out_indices;
            tmp_hashes.reserve(hashes.size() + rebuilding_data_size);
            tmp_indices.reserve(hashes.size() + rebuilding_data_size);
            out_hashes.reserve(hashes.size() + rebuilding_data_size);
            out_indices.reserve(hashes.size() + rebuilding_data_size);

            if (hashes.size() != 0) {
                // Move data to temporary vector for sorting.
                for (size_t i=SEGMENT_SIZE; i < hashes.size()-SEGMENT_SIZE; i++) {
                    tmp_hashes.push_back(hashes[i]);
                    tmp_indices.push_back(indices[i]);
                }
            }
            for (auto & rebuilding_data : parallel_rebuilding_data) {
                for (auto pair : rebuilding_data) {
                    tmp_indices.push_back(pair.first);
                    tmp_hashes.push_back(pair.second);
                }
            }
            
            puffinn::sort_hashes_pairs_24(
                tmp_hashes,
                out_hashes,
                tmp_indices,
                out_indices
            );

            // Pad with SEGMENT_SIZE values on each size to remove need for bounds check.
            hashes.clear();
            hashes.reserve(out_hashes.size() + 2*SEGMENT_SIZE);
            indices.clear();
            indices.reserve(out_hashes.size() + 2*SEGMENT_SIZE);

            for (int i=0; i < SEGMENT_SIZE; i++) {
                hashes.push_back(IMPOSSIBLE_PREFIX);
                indices.push_back(0);
            }
            for (size_t i = 0; i < out_hashes.size(); i++) {
                indices.push_back(out_indices[i]);
                hashes.push_back(out_hashes[i]);
            }
            for (int i=0; i < SEGMENT_SIZE; i++) {
                hashes.push_back(IMPOSSIBLE_PREFIX);
                indices.push_back(0);
            }

            // Build prefix_index data structure.
            // Index of the first occurence of the prefix
            uint32_t idx = 0;
            for (unsigned int prefix=0; prefix < (1u << PREFIX_INDEX_BITS); prefix++) {
                while (
                    idx < rebuilding_data_size &&
                    (hashes[SEGMENT_SIZE+idx] >> (hash_length-PREFIX_INDEX_BITS)) < prefix
                ) {
                    idx++;
                }
                prefix_index[prefix] = SEGMENT_SIZE+idx;
            }
            prefix_index[1 << PREFIX_INDEX_BITS] = SEGMENT_SIZE+rebuilding_data_size;

            for (auto & rd : parallel_rebuilding_data) {
                rd.clear();
                rd.shrink_to_fit();
            }

        }

        // Construct a query object to search for the nearest neighbors of the given vector.
        PrefixMapQuery create_query(LshDatatype hash) const {
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
