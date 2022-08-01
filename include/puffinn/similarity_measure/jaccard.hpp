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

        static size_t intersection_size_gallop(Format::Type* small_ptr, Format::Type* large_ptr) {
            size_t size_small = small_ptr->size();
            size_t size_large = large_ptr->size();
            if (size_small > size_large) {
                // Swap the arguments
                return intersection_size_gallop(large_ptr, small_ptr);
            }
            auto& small = *small_ptr;
            auto& large = *large_ptr;
            size_t size = 0;
            size_t small_idx = 0;
            size_t large_idx = 0;
            size_t index = 0;
            while(small_idx < size_small && large_idx < size_large) {
                auto target = small[small_idx];
                size_t diff = 1;
                while (large_idx + diff < size_large && large[large_idx + diff] < target) {
                    diff *= 2;
                }
                size_t end = large_idx + diff;
                if (end > size_large) {
                    end = size_large;
                }

                {
                    size_t n = end - large_idx;
                    if (n == 0) {
                        return size;
                    }
                    auto base = large_idx;
                    while (n > 1) {
                        size_t half = n >> 1;
                        base = (large[base + half] < target) ? base + half : base;
                        n -= half;
                    }
                    index = (large[base] < target) ? base + 1 : base;
                }
                if ((index < size_large) && (large[index] == target)) {
                    size++;
                }
                small_idx++;
                large_idx = index;
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
