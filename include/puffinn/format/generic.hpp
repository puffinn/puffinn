#pragma once

#include <istream>
#include <memory>
#include <ostream>

namespace puffinn {
    // Both the dimensionality of the input vectors and the
    // size of the array they are actually stored in.
    template <typename T>
    struct DatasetDescription {
        typename T::Args args;
        unsigned int storage_len;

        DatasetDescription() = default;
        DatasetDescription(std::istream& in) {
            T::deserialize_args(in, &args);
            in.read(reinterpret_cast<char*>(&storage_len), sizeof(unsigned int));
        }

        void serialize(std::ostream& out) const {
            T::serialize_args(out, args);
            out.write(reinterpret_cast<const char*>(&storage_len), sizeof(unsigned int));
        }
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

    // Aligns data by allocating additional space.
    template <typename T>
    class AlignedStorage {
        void* raw_mem;
        typename T::Type* aligned;
        size_t len;

        void reset() {
            raw_mem = nullptr;
            aligned = nullptr;
            len = 0;
        }

    public:
        AlignedStorage() {
            reset();
        }

        AlignedStorage(size_t len)
          : len(len)
        {
            size_t buffer_len = len*sizeof(typename T::Type)+T::ALIGNMENT;

            raw_mem = operator new(buffer_len);
            void* raw_aligned = raw_mem;
            if (T::ALIGNMENT != 0) {
                std::align(
                    T::ALIGNMENT,
                    len*sizeof(typename T::Type),
                    raw_aligned,
                    buffer_len);
            }

            aligned = static_cast<typename T::Type* const>(raw_aligned);
            for (size_t i=0; i < len; i++) {
                new(aligned+i) typename T::Type();
            }
        }

        AlignedStorage(AlignedStorage&& other)
          : raw_mem(other.raw_mem),
            aligned(other.aligned),
            len(other.len)
        {
            other.reset();
        }

        AlignedStorage& operator=(AlignedStorage&& rhs) {
            if (this != &rhs) {
                raw_mem = rhs.raw_mem;
                aligned = rhs.aligned;
                len = rhs.len;
                rhs.reset();
            }
            return *this;
        }

        ~AlignedStorage() {
            for (size_t i=0; i < len; i++) {
                T::free(aligned[i]);
            }
            operator delete(raw_mem);
        }

        typename T::Type* get() const {
            return aligned;
        }
    };

    // Allocate a number of vectors of a specific format.
    template <typename T>
    AlignedStorage<T> allocate_storage(
        size_t vector_count,
        unsigned int padded_dimensions
    ) {
        return AlignedStorage<T>(vector_count*padded_dimensions);
    }

    // Convert the input type to the internal format.
    template <typename T, typename U>
    AlignedStorage<T> to_stored_type(
        const U& input,
        DatasetDescription<T> desc
    ) {
        auto storage = allocate_storage<T>(1, desc.storage_len);
        T::store(input, storage.get(), desc);
        return storage;
    }

    template <typename F, typename T>
    T convert_stored_type(typename F::Type*, DatasetDescription<F>);

}
