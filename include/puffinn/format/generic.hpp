#pragma once

#include <memory>

namespace puffinn {
    // Both the dimensionality of the input vectors and the
    // size of the array they are actually stored in.
    template <typename T>
    struct DatasetDescription {
        typename T::Args args;
        unsigned int storage_len;
    };

    // Round up the given value to the first value that is a multiple of the second argument.
    constexpr unsigned int ceil_to_multiple(unsigned int val, unsigned int mult) {
        return (val+mult-1)/mult*mult;
    }

    // Calculate the amount of padding that is required by the format.
    template <typename T>
    unsigned int pad_dimensions(unsigned int dimensions) {
        if (T::ALIGNMENT == 0) {
            // No need for padding when no aligment requirements.
            return dimensions;
        }
        return ceil_to_multiple(dimensions, T::ALIGNMENT/sizeof(typename T::Type));
    }

    // Allocate a number of vectors of a specific format.
    template <typename T>
    std::unique_ptr<typename T::Type, decltype(free)*> allocate_storage(
        size_t vector_count,
        unsigned int padded_dimensions
    ) {
        size_t bytes = vector_count*padded_dimensions*sizeof(typename T::Type);
        // alignment 0 means that it does not matter
        void* raw_storage = T::ALIGNMENT == 0 ? malloc(bytes) : aligned_alloc(T::ALIGNMENT, bytes);
        auto storage = static_cast<typename T::Type*>(raw_storage);
        // Ensure data is in a valid state by default constructing them.
        for (size_t i=0; i < vector_count*padded_dimensions; i++) {
            new(storage+i) typename T::Type();
        }
        return std::unique_ptr<typename T::Type, decltype(free)*>(storage, free);
    }

    // Convert the input type to the internal format.
    template <typename T, typename U>
    std::unique_ptr<typename T::Type, decltype(free)*> to_stored_type(
        const U& input,
        DatasetDescription<T> desc
    ) {
        auto storage = allocate_storage<T>(1, desc.storage_len);
        T::store(input, storage.get(), desc);
        return storage;
    }
}
