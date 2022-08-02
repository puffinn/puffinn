#pragma once

#include "puffinn/format/set.hpp"

namespace puffinn {
    class MinHash;
    class MinHash1Bit;
    
    /// Measures the Jaccard Similarity between two sets.
    ///
    /// This is defined as the size of the intersection divided by the size of the union.
    /// The supported LSH families are ``MinHash`` and ``MinHash1Bit``. 
    struct JaccardSimilarity {
        using Format = SetFormat; 
        using DefaultSketch = MinHash1Bit;
        using DefaultHash = MinHash;

        /// Returns the index of the key in the given range (begin inclusive, end exclusive) of the array
        static int32_t binary_search(Format::Type * vals_ptr, size_t begin, size_t end, uint32_t key) {
            if ((end > 0) && ((*vals_ptr)[end - 1] < key)) {
                return -end -1;
            }
            size_t low = begin;
            size_t high = end - 1;
            while (low <= high) {
                size_t middleIndex = (low + high) >> 1;
                size_t middleValue = (*vals_ptr)[middleIndex];
                if (middleValue < key) {
                    low = middleIndex + 1;
                } else if (middleValue > key) {
                    high = middleIndex - 1;
                } else {
                    return middleIndex;
                }
            }
            return -(low+1);
        }

        /// Starting from `begin`, returns the first index in the given vector such that
        /// the corresponding value is >= the given key
        static size_t gallop(Format::Type* vec_ptr, uint32_t key, size_t begin, size_t end) {
            size_t range_end = end;
            auto& vec = *vec_ptr;
            size_t offset = 1;
            // Jump with geometrically increasing windows until we find the window
            // containing the sought key.
            while (begin + offset < end && vec[begin + offset] < key) {
                begin += offset;
                offset *= 2;
            }
            if (begin + offset < end) {
                end = begin + offset + 1;
            }

            // Binary search the key in the range
            size_t low = begin;
            size_t high = end - 1;
            while (low <= high) {
                size_t middleIndex = (low + high) >> 1;
                size_t middleValue = vec[middleIndex];
                if (middleValue < key) {
                    low = middleIndex + 1;
                } else if (middleValue > key) {
                    high = middleIndex - 1;
                } else {
                    return middleIndex;
                }
            }
            return low;
        }

        static size_t intersection_size_gallop(Format::Type* lhs_ptr, Format::Type* rhs_ptr) {
            size_t size = 0;
            auto& lhs = *lhs_ptr;
            auto& rhs = *rhs_ptr;

            size_t lhs_idx = 0;
            size_t rhs_idx = 0;
            size_t lhs_size = lhs.size();
            size_t rhs_size = rhs.size();

            while (lhs_idx < lhs_size && rhs_idx < rhs_size) {
                if (lhs[lhs_idx] == rhs[rhs_idx]) {
                    size++;
                    lhs_idx++;
                    rhs_idx++;
                } else if (lhs[lhs_idx] < rhs[rhs_idx]) {
                    lhs_idx = gallop(lhs_ptr, rhs[rhs_idx], lhs_idx, lhs_size);
                } else {
                    rhs_idx = gallop(rhs_ptr, lhs[lhs_idx], rhs_idx, rhs_size);
                }
            }

            return size;
        }

        static size_t intersection_size_linear(Format::Type* lhs_ptr, Format::Type* rhs_ptr) {
            auto& lhs = *lhs_ptr;
            auto& rhs = *rhs_ptr;
            int intersection_size = 0;
            size_t lhs_idx = 0;
            size_t rhs_idx = 0;
            while (lhs_idx < lhs.size() && rhs_idx < rhs.size()) {
                if (lhs[lhs_idx] == rhs[rhs_idx]) {
                    intersection_size++;
                    lhs_idx++;
                    rhs_idx++;
                } else if(lhs[lhs_idx] < rhs[rhs_idx]) {
                    lhs_idx++;
                } else {
                    rhs_idx++;
                }
            }
            return intersection_size;
        }

        static float compute_similarity(Format::Type* lhs_ptr, Format::Type* rhs_ptr, DatasetDescription<Format>) {
            return compute_similarity_linear(lhs_ptr, rhs_ptr);
        }

        static float compute_similarity_linear(Format::Type* lhs_ptr, Format::Type* rhs_ptr) {
            auto& lhs = *lhs_ptr;
            auto& rhs = *rhs_ptr;
            float intersection = intersection_size_linear(lhs_ptr, rhs_ptr);
            auto divisor = lhs.size()+rhs.size()-intersection;
            if (divisor == 0) {
                return 0;
            } else {
                return intersection/divisor;
            }
        }

        static float compute_similarity_gallop(Format::Type* lhs_ptr, Format::Type* rhs_ptr) {
            float intersection = intersection_size_gallop(lhs_ptr, rhs_ptr);
            auto divisor = lhs_ptr->size()+rhs_ptr->size()-intersection;
            if (divisor == 0) {
                return 0;
            } else {
                return intersection/divisor;
            }
        }

    };
}

#include "puffinn/hash/minhash.hpp"
