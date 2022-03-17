#pragma once

#include <chrono>
#include <cstdint>
#include <random>

namespace puffinn {
    // Number of bits used in filtering sketches.
    const static unsigned int NUM_FILTER_HASHBITS = 64;
    using FilterLshDatatype = uint64_t;

    // Number of bits used in hashes.
    const static unsigned int MAX_HASHBITS = 24;
    // The hash_pool concatenates hashes into a type twice as large to avoid overflow errors.
    // TODO: Check how to avoid this "twice as large": using 64 bits to store 24-bits hash values.
    // To me it seems that when inserting into a prefix-map, 64 bits are automatically 
    // truncated to `LshDatatype` when push_back is called on `rebuilding_data`.
    using LshDatatype = uint32_t;

    // std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    // for reproducibility, fix the seed of the random number generator
    std::default_random_engine generator(1234);

    void reset_random_generator() {
        generator = std::default_random_engine(1234);
    }

    // Retrieve the default random engine, seeded once by the system clock.
    std::default_random_engine& get_default_random_generator() {
        return generator;
    }

    #if defined(__GNUC__)
        #define popcountll __builtin_popcountll
    # elif defined(_MSC_VER)
        #define popcountll __popcnt64
    #else
        // Code from paper:
        // Faster Population Counts Using AVX2 Instructions
        // Wojciech Mula, Nathan Kurz and Daniel Lemire
        uint64_t popcountll(uint64_t x) {
            uint64_t c1 = 0x5555555555555555llu;
            uint64_t c2 = 0x3333333333333333llu;
            uint64_t c4 = 0x0F0F0F0F0F0F0F0Fllu;

            x -= (x >> 1) & c1;
            x = ((x >> 2) & c2) + (x & c2);
            x = (x + (x >> 4)) & c4;
            x *= 0x0101010101010101llu;
            return x >> 56;
        }
    #endif

    #if defined(__GNUC__)
        #define prefetch_addr __builtin_prefetch
    #else
        void prefetch_addr(void*) { /* noop */ }
    #endif
 }
