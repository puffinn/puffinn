#pragma once

#include "puffinn/format/generic.hpp"
#include "puffinn/typedefs.hpp"

#include <cstring>
#include <memory>

namespace puffinn {
    const unsigned int DEFAULT_CAPACITY = 100;
    const float EXPANSION_FACTOR = 1.5;

    // The container for all inserted vectors.
    // The data is stored according to the given format.
    template <typename T>
    class Dataset {
        // Number of dimensions of inserted vectors.
        unsigned int dimensions;
        // Number of dimensions of stored vectors, which may be padded to
        // more easily map to simd instructions.
        unsigned int padded_dimensions;
        // Number of inserted vectors.
        unsigned int inserted_vectors;
        // Maximal number of inserted vectors.
        unsigned int capacity;
        // Inserted vectors, aligned to the vector alignment.
        std::unique_ptr<typename T::Type, decltype(free)*> data;

    public:
        // Create an empty storage for vectors with the given number of dimensions.
        Dataset(unsigned int dimensions) : Dataset(dimensions, DEFAULT_CAPACITY)
        {
        }

        // Create an empty storage for vectors with the given number of dimensions.
        // Allocates enough space for the given number of vectors before needing to reallocate.
        Dataset(unsigned int dimensions, unsigned int capacity)
          : dimensions(dimensions),
            padded_dimensions(pad_dimensions<T>(dimensions)),
            inserted_vectors(0),
            capacity(capacity),
            data(allocate_storage<T>(capacity, padded_dimensions))
        {
        }

        // Access the vector at the given position.
        typename T::Type* operator[](unsigned int idx) const {
            return &data.get()[idx*padded_dimensions];
        }

        // Retrieve the number of dimensions of vectors inserted into this dataset,
        // as well as the number of dimensions they are stored with.
        DatasetDimensions get_dimensions() const {
            DatasetDimensions res;
            res.actual = dimensions;
            res.padded = padded_dimensions;
            return res;
        }

        // Retrieve the number of inserted vectors.
        unsigned int get_size() const {
            return inserted_vectors;
        }

        // Insert a vector.
        template <typename U>
        void insert(const U& vec) {
            if (inserted_vectors == capacity) {
                unsigned int new_capacity = std::ceil(capacity*EXPANSION_FACTOR);
                auto new_data = allocate_storage<T>(new_capacity, padded_dimensions);
                for (size_t i=0; i < capacity*padded_dimensions; i++) {
                    new_data.get()[i] = std::move(data.get()[i]);
                }
                data = std::move(new_data);
                capacity = new_capacity;
            }
            T::store(
                vec,
                &data.get()[inserted_vectors*padded_dimensions],
                get_dimensions());
            inserted_vectors++;
        }

        // Retrieve the capacity of the dataset
        unsigned int get_capacity() const {
            return capacity;
        }

        // Remove all points from the dataset.
        void clear() {
            inserted_vectors = 0;
        }
    };
}
