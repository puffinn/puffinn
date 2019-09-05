#pragma once

#include "puffinn/hash_source/hash_source.hpp"
#include "puffinn/hash_source/independent.hpp"
#include "puffinn/hash_source/pool.hpp"
#include "puffinn/hash_source/tensor.hpp"

namespace puffinn {
    template <typename T>
    static std::unique_ptr<HashSourceArgs<T>> deserialize_hash_args(std::istream& in) {
        HashSourceType type;
        in.read(reinterpret_cast<char*>(&type), sizeof(HashSourceType));
        switch (type) {
            case HashSourceType::Independent:
                return std::make_unique<IndependentHashArgs<T>>(in);
            case HashSourceType::Pool:
                return std::make_unique<HashPoolArgs<T>>(in);
            case HashSourceType::Tensor:
                return std::make_unique<TensoredHashArgs<T>>(in);
            default:
                throw std::invalid_argument("hash source type");
        }
    }
}
