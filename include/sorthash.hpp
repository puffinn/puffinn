#pragma once

#include "puffinn/typedefs.hpp"
#include <algorithm>

namespace puffinn {

#define _0(x)	(x & 0xFF)
#define _1(x)	(x >> 8 & 0xFF)
#define _2(x)	(x >> 16 & 0xFF)

// This macro is useful for debugging purposes
#define print_bytes(arr) \
    for (size_t i = 0; i < arr.size(); i++) { \
        const uint32_t hi = arr[i]; \
        printf("[%d]  %d  |  %d  |  %d\n", i, _0(hi), _1(hi), _2(hi)); \
    } \
    printf("\n\n");

//! Sort the given vector of hash values, with n bytes of auxiliary space, using MSB Radix Sort
//! with three passes over the data. Assumes that the hash values are stored in 
//! the 24 least significant bits of each 32 bit integer.
void sort_hashes_24(std::vector<uint32_t> & hashes, std::vector<uint32_t> & out) {
    const size_t n = hashes.size();
    const size_t n_bytes = 256;
    out.clear();
    out.resize(n, 0);

    // Histograms on the stack
    uint32_t b0[n_bytes], b1[n_bytes], b2[n_bytes];
    for (size_t i = 0; i < n_bytes; i++) {
        b0[i] = 0;
        b1[i] = 0;
        b2[i] = 0;
    }

    // One-pass histogram computation
    for (size_t i = 0; i < n; i++) {
        const uint32_t hi = hashes[i];
        b0[_0(hi)]++;
        b1[_1(hi)]++;
        b2[_2(hi)]++;
    }

    // Cumulative sum of the histograms, which then keep track 
    // of the write head position for each byte
    {
        uint32_t tsum = 0; // Re-set and reused for all three histograms
        uint32_t
            sum0 = 0,
            sum1 = 0,
            sum2 = 0;

        for (size_t i = 0; i < n_bytes; i++) {
            tsum = sum0 + b0[i];
            b0[i] = sum0;
            sum0 = tsum;

            tsum = sum1 + b1[i];
            b1[i] = sum1;
            sum1 = tsum;

            tsum = sum2 + b2[i];
            b2[i] = sum2;
            sum2 = tsum;
        }
    }

    // First sorting pass
    for (size_t i = 0; i < n; i++) {
        const uint32_t hi = hashes[i];
        const uint32_t pos = _0(hi);
        out[b0[pos]++] = hi;
    }

    // Second sorting pass
    for (size_t i = 0; i < n; i++) {
        const uint32_t hi = out[i];
        const uint32_t pos = _1(hi);
        hashes[b1[pos]++] = hi;
    }
    
    // Third sorting pass
    for (size_t i = 0; i < n; i++) {
        const uint32_t hi = hashes[i];
        const uint32_t pos = _2(hi);
        out[b2[pos]++] = hi;
    }
}

} // namespace puffinn
