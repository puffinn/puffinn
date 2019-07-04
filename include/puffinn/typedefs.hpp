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
    using LshDatatype = uint32_t;

    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());

    // Retrieve the default random engine, seeded once by the system clock.
    std::default_random_engine& get_default_random_generator() {
        return generator;
    }
}
