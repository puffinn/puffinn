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

#define do_pass_simple1(arr_in, arr_out, bx, get_byte) \
    for (size_t i = 0; i < n; i++) {           \
        const uint32_t hi = arr_in[i];     \
        const uint32_t pos = get_byte(hi);       \
        arr_out[bx[pos]++] = hi;           \
    }

#define do_pass_unrolled1(arr_in, arr_out, bx, get_byte)       \
    {                                                         \
    size_t off = 0;                                           \
    for (; off + 4 < n; off += 4) {                           \
        const uint32_t hi0 = arr_in[off    ];                 \
        const uint32_t hi1 = arr_in[off + 1];                 \
        const uint32_t hi2 = arr_in[off + 2];                 \
        const uint32_t hi3 = arr_in[off + 3];                 \
                                                              \
        const uint32_t pos0 = get_byte(hi0);                  \
        const uint32_t pos1 = get_byte(hi1);                  \
        const uint32_t pos2 = get_byte(hi2);                  \
        const uint32_t pos3 = get_byte(hi3);                  \
                                                              \
        arr_out[bx[pos0]++] = hi0;                            \
        arr_out[bx[pos1]++] = hi1;                            \
        arr_out[bx[pos2]++] = hi2;                            \
        arr_out[bx[pos3]++] = hi3;                            \
    }                                                         \
    for (; off < n; off++) {                                  \
        const uint32_t hi = arr_in[off];                      \
        const uint32_t pos = get_byte(hi);                    \
        arr_out[bx[pos]++] = hi;                              \
    }                                                         \
    }              

#define do_pass1 do_pass_unrolled1

//! Sort the given vector of hash values, with n bytes of auxiliary space, using adix Sort
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
    do_pass1(hashes, out, b0, _0);

    // Second sorting pass
    do_pass1(out, hashes, b1, _1);

    // Third sorting pass
    do_pass1(hashes, out, b2, _2);
}

#define do_pass_simple(arr_in, arr_out, idx_in, idx_out, bx, get_byte) \
    for (size_t i = 0; i < n; i++) {           \
        const uint32_t hi = arr_in[i];     \
        const uint32_t pos = get_byte(hi);       \
        const uint32_t t = bx[pos]++;   \
        arr_out[t] = hi;           \
        idx_out[t] = idx_in[i]; \
    }

#define do_pass do_pass_simple

//! Sort the given vector of hash values, along with the corresponding vector of 
//! identifiers, with n bytes of auxiliary space, using Radix Sort
//! with three passes over the data. Assumes that the hash values are stored in 
//! the 24 least significant bits of each 32 bit integer.
//!
//! In this sort routine, the indices provided in the `idx_in` argument are considered
//! as forming a pair with the hashes in `hashes_in`, and will be sorted along with them
//! as if we were sorting an array of std::pair using the hash as key.
void sort_hashes_pairs_24(
    std::vector<uint32_t> & hashes_in,
    std::vector<uint32_t> & hashes_out,
    std::vector<uint32_t> & idx_in,
    std::vector<uint32_t> & idx_out
) {
    const size_t n = hashes_in.size();
    const size_t n_bytes = 256;
    hashes_out.clear();
    hashes_out.resize(n, 0);
    idx_out.clear();
    idx_out.resize(n, 0);

    // Histograms on the stack
    uint32_t b0[n_bytes], b1[n_bytes], b2[n_bytes];
    for (size_t i = 0; i < n_bytes; i++) {
        b0[i] = 0;
        b1[i] = 0;
        b2[i] = 0;
    }

    // One-pass histogram computation
    for (size_t i = 0; i < n; i++) {
        const uint32_t hi = hashes_in[i];
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
    do_pass(hashes_in, hashes_out, idx_in, idx_out, b0, _0);

    // Second sorting pass
    do_pass(hashes_out, hashes_in, idx_out, idx_in, b1, _1);

    // Third sorting pass
    do_pass(hashes_in, hashes_out, idx_in, idx_out, b2, _2);
}


} // namespace puffinn
