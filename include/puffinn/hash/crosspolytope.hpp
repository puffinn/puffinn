#pragma once

#include "puffinn/dataset.hpp"
#include "external/ffht/fht_header_only.h"
#include "puffinn/format/unit_vector.hpp"
#include "puffinn/math.hpp"
#include "puffinn/similarity_measure/cosine.hpp"

namespace puffinn {
    struct CrossPolytopeCollisionEstimates {
        std::vector<std::vector<float>> probabilities;
        float eps;

        CrossPolytopeCollisionEstimates() {}

        CrossPolytopeCollisionEstimates(
            unsigned int dimensions,
            unsigned int num_repetitions,
            float eps
        )
          : eps(eps)
        {
            // adapted from https://bitbucket.org/tobc/knn_code/src/master/crosspolytopefamily.h
            std::normal_distribution<double> standard_normal(0,1);
            auto& rng = get_default_random_generator();

            auto log_dimensions = ceil_log(dimensions);
            // Number of collisions for each number of used bits
            std::vector<int> collisions(log_dimensions+2);
            probabilities = std::vector<std::vector<float>>(log_dimensions+2);

            double alpha = -1;
            //foreach [alpha, alpha+eps) segment
            while(alpha <= 1) {
                for (auto& v : collisions) { v = 0; }

                for(uint32_t i = 0; i < num_repetitions; i++) {
                    // length = dimensions
                    // x = (1, 0, ..., 0)
                    // y = (alpha, (1-alpha^2)^(1/2), ..., 0)

                    // The hash value so far.
                    uint32_t hash_x = 0;
                    uint32_t hash_y = 0;
                    // Absolute value of highest value seen.
                    double v_x = 0;
                    double v_y = 0;

                    // Compute a random rotation of x and y using the matrix z
                    // [ [ z_1_0, z_2_0 ],
                    //   [ z_1_1, z_2_1 ],
                    //   [ z_1_j, z_2_j ] ]
                    for(uint32_t j = 0; j < dimensions; j++) {
                        double z_1 = standard_normal(rng);
                        double z_2 = standard_normal(rng);
                        // calculate z*x[j] and find the index with the highest value
                        if(abs(z_1) > v_x) {
                            v_x = abs(z_1);
                            hash_x = j;
                            if (z_1 < 0) { hash_x |= (1 << log_dimensions); }
                        }
                        // do the same for z*y[j]
                        double h_y = alpha*z_1 + pow(1 - pow(alpha, 2), 0.5)*z_2;
                        if(abs(h_y) > v_y) {
                            v_y = abs(h_y);
                            hash_y = j;
                            if (h_y < 0) { hash_y |= (1 << log_dimensions); }
                        }
                    }
                    for (unsigned int used_bits = 0; used_bits <= log_dimensions+1; used_bits++) {
                        auto shift = log_dimensions+1-used_bits;
                        collisions[used_bits] += (hash_x >> shift) == (hash_y >> shift);
                    }
                }
                for (unsigned int used_bits = 0; used_bits <= log_dimensions+1; used_bits++) {
                    auto prob = static_cast<float>(collisions[used_bits])/num_repetitions;
                    probabilities[used_bits].push_back(prob);
                }
                // eps refers to the number of segments between 0 and 1, but the estimation
                // works in segments from -1 to 1.
                alpha += 2*eps;
            }
        }

        CrossPolytopeCollisionEstimates(std::istream& in) {
            size_t d1;
            in.read(reinterpret_cast<char*>(&d1), sizeof(size_t));

            for (size_t i=0; i < d1; i++) {
                size_t d2;
                in.read(reinterpret_cast<char*>(&d2), sizeof(size_t));

                probabilities.emplace_back(d2);
                in.read(reinterpret_cast<char*>(&probabilities[i][0]), d2*sizeof(float));
            }
            in.read(reinterpret_cast<char*>(&eps), sizeof(float));
        }

        void serialize(std::ostream& out) const {
            size_t d1 = probabilities.size();
            out.write(reinterpret_cast<char*>(&d1), sizeof(size_t));

            for (size_t i=0; i < d1; i++) {
                size_t d2 = probabilities[i].size();
                out.write(reinterpret_cast<char*>(&d2), sizeof(size_t));
                out.write(reinterpret_cast<const char*>(&probabilities[i][0]), d2*sizeof(float));
            }
            out.write(reinterpret_cast<const char*>(&eps), sizeof(float));
        }

        float get_collision_probability(float sim, int_fast8_t num_bits) const {
            return probabilities[num_bits][(size_t)(sim/eps)];
        }
    };

    class FHTCrossPolytopeHashFunction {
        int dimensions;
        int log_dimensions;
        unsigned int num_rotations;
        // Random +-1 diagonal matrix for each rotation in each application of cross-polytope.
        // Hash idx * num_rotations * dimensions as power of 2
        std::vector<int8_t> random_signs;

        // Calculate a unique value depending on which axis is closest to the given floating point
        // vector.
        LshDatatype encode_closest_axis(float* vec) const {
            int res = 0;
            float max_sim = 0;
            for (int i = 0; i < (1 << log_dimensions); i++) {
                if (vec[i] > max_sim) {
                    res = i;
                    max_sim = vec[i];
                } else if (-vec[i] > max_sim) {
                    res = i+(1 << log_dimensions);
                    max_sim = -vec[i];
                }
            }
            return res;
        }

    public:
        // Create a cross polytope hasher using the given number of pseudorandom rotations
        // using hadamard transforms.
        FHTCrossPolytopeHashFunction(
            DatasetDescription<UnitVectorFormat> dataset,
            unsigned int num_rotations
        )
          : dimensions(dataset.args),
            num_rotations(num_rotations)
        {
            log_dimensions = ceil_log(dimensions);

            int random_signs_len = num_rotations*(1 << log_dimensions);
            random_signs.reserve(random_signs_len);

            std::uniform_int_distribution<int_fast32_t> sign_distribution(0, 1);
            auto& generator = get_default_random_generator();
            for (int i=0; i < random_signs_len; i++) {
                random_signs.push_back(sign_distribution(generator)*2-1);
            }
        }

        FHTCrossPolytopeHashFunction(std::istream& in) {
            in.read(reinterpret_cast<char*>(&dimensions), sizeof(int));
            in.read(reinterpret_cast<char*>(&log_dimensions), sizeof(int));
            in.read(reinterpret_cast<char*>(&num_rotations), sizeof(unsigned int));

            int signs_len = num_rotations*(1 << log_dimensions);
            random_signs = std::vector<int8_t>(signs_len);
            in.read(reinterpret_cast<char*>(&random_signs[0]), signs_len*sizeof(int8_t));
        }

        void serialize(std::ostream& out) const {
            out.write(reinterpret_cast<const char*>(&dimensions), sizeof(int));
            out.write(reinterpret_cast<const char*>(&log_dimensions), sizeof(int));
            out.write(reinterpret_cast<const char*>(&num_rotations), sizeof(unsigned int));

            out.write(reinterpret_cast<const char*>(&random_signs[0]), random_signs.size()*sizeof(int8_t));
        }

        // Hash the given vector
        LshDatatype operator()(int16_t* vec) const {
            float rotated_vec[1 << log_dimensions];

            // Reset rotation vec
            for (int i=0; i<dimensions; i++) {
                rotated_vec[i] = UnitVectorFormat::from_16bit_fixed_point(vec[i]);
            }
            for (int i=dimensions; i < (1 << log_dimensions); i++) {
                rotated_vec[i] = 0.0f;
            }

            for (unsigned int rotation = 0; rotation < num_rotations; rotation++) {
                // Multiply by a diagonal +-1 matrix.
                int sign_idx = rotation*(1 << log_dimensions);
                for (int i=0; i < (1 << log_dimensions); i++) {
                    rotated_vec[i] *= random_signs[sign_idx+i];
                }
                // Apply the fast hadamard transform
                fht(rotated_vec, log_dimensions);
            }

            return encode_closest_axis(rotated_vec);
        }
    };

    /// Arguments for the fast-hadamard cross-polytope LSH.
    struct FHTCrossPolytopeArgs {
        /// Number of iterations of the fast-hadamard transform. 
        int num_rotations;
        /// Number of samples used to estimate collision probabilities.
        unsigned int estimation_repetitions;
        /// Granularity of collision probability estimation.
        float estimation_eps;

        constexpr FHTCrossPolytopeArgs()
            : num_rotations(3),
              estimation_repetitions(1000),
              estimation_eps(5e-3)
        {
        }

        FHTCrossPolytopeArgs(std::istream& in) {
            in.read(reinterpret_cast<char*>(&num_rotations), sizeof(int));
            in.read(reinterpret_cast<char*>(&estimation_repetitions), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&estimation_eps), sizeof(float));
        }

        void serialize(std::ostream& out) const {
            out.write(reinterpret_cast<const char*>(&num_rotations), sizeof(int));
            out.write(reinterpret_cast<const char*>(&estimation_repetitions), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&estimation_eps), sizeof(float));
        }

        uint64_t memory_usage(DatasetDescription<UnitVectorFormat> dataset) const {
            return sizeof(FHTCrossPolytopeHashFunction)
            + num_rotations*(1 << ceil_log(dataset.args))*sizeof(int8_t);
        }

        void set_no_preprocessing() {
            estimation_repetitions = 0;
            estimation_eps = 2.0;
        }
    };

    /// An implementation of cross-polytope LSH using fast-hadamard transforms.
    /// 
    /// See ``CrossPolytopeHash`` for a description of the hash function.
    ///
    /// Using repeated applications of random +/-1 diagonal matrices and hadamard transforms,
    /// a pseudo-random rotation can be calculated more efficiently than using a random rotation. 
    /// In practice, using three applications of each transform 
    /// gives hashes of similar quality as the ones produced using a true random rotation.
    class FHTCrossPolytopeHash {
    public:
        using Args = FHTCrossPolytopeArgs;
        using Sim = CosineSimilarity;
        using Function = FHTCrossPolytopeHashFunction;

    private:
        DatasetDescription<UnitVectorFormat> dataset;
        Args args;
        CrossPolytopeCollisionEstimates estimates;

    public:
        FHTCrossPolytopeHash(
            DatasetDescription<UnitVectorFormat> dataset,
            Args args
        )
          : dataset(dataset),
            args(args),
            estimates(
                (1 << ceil_log(dataset.args)),
                args.estimation_repetitions,
                args.estimation_eps)
        {
        }

        FHTCrossPolytopeHash(std::istream& in)
          : dataset(in),
            args(in),
            estimates(in)
        {
        }

        void serialize(std::ostream& out) const {
            dataset.serialize(out);
            args.serialize(out);
            estimates.serialize(out);
        }

        FHTCrossPolytopeHashFunction sample() {
            return FHTCrossPolytopeHashFunction(dataset, args.num_rotations);
        }

        unsigned int bits_per_function() {
            return ceil_log(dataset.args)+1;
        }
 
        float collision_probability(
            float similarity,
            int_fast8_t num_bits
        ) const {
            return estimates.get_collision_probability(similarity, num_bits);
        }
    };

    class CrossPolytopeHashFunction {
        unsigned int dimensions;
        unsigned int padded_dimensions;
        std::unique_ptr<int16_t, decltype(free)*> random_matrix;

    public:
        CrossPolytopeHashFunction(DatasetDescription<UnitVectorFormat> dataset)
          : dimensions(dataset.args),
            padded_dimensions(dataset.storage_len),
            random_matrix(
                allocate_storage<UnitVectorFormat>(
                    1 << ceil_log(dimensions), 
                    dataset.storage_len))
        {
            unsigned int matrix_size = (1 << ceil_log(dimensions));

            for (unsigned int dim=0; dim < matrix_size; dim++) {
                auto vec = UnitVectorFormat::generate_random(dimensions);
                UnitVectorFormat::store(
                    vec,
                    &random_matrix.get()[dim*padded_dimensions],
                    dataset);
            }
        }

        CrossPolytopeHashFunction(std::istream& in)
          : random_matrix(nullptr, &free)
        {
            in.read(reinterpret_cast<char*>(&dimensions), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&padded_dimensions), sizeof(unsigned int));
            random_matrix = allocate_storage<UnitVectorFormat>(
                (1 << ceil_log(dimensions)),
                padded_dimensions);
            auto matrix_len = (1 << ceil_log(dimensions))*padded_dimensions;
            in.read(reinterpret_cast<char*>(random_matrix.get()), matrix_len*sizeof(int16_t));
        }

        void serialize(std::ostream& out) const {
            out.write(reinterpret_cast<const char*>(&dimensions), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&padded_dimensions), sizeof(unsigned int));

            auto matrix_len = (1 << ceil_log(dimensions))*padded_dimensions;
            out.write(reinterpret_cast<const char*>(random_matrix.get()), matrix_len*sizeof(int16_t));
        }

        LshDatatype operator()(int16_t* vec) const {
            LshDatatype res = 0;
            uint16_t max_abs_dot = 0;
            for (unsigned int i=0; i<(1u << ceil_log(dimensions)); i++) {
                auto matrix_row = &random_matrix.get()[i*padded_dimensions];
                // dot product
                auto rotated_i = dot_product_i16(vec, matrix_row, dimensions);
                if (rotated_i > max_abs_dot) {
                    max_abs_dot = rotated_i;
                    res = i;
                } else if (-rotated_i > max_abs_dot) {
                    max_abs_dot = -rotated_i;
                    res = i+(1 << ceil_log(dimensions));
                }
            }
            return res;
        }
    };

    /// Arguments for the cross-polytope LSH.
    struct CrossPolytopeArgs {
        /// Number of samples used to estimate collision probabilities.
        unsigned int estimation_repetitions;
        /// Granularity of collision probability estimation.
        float estimation_eps;

        constexpr CrossPolytopeArgs()
          : estimation_repetitions(1000),
            estimation_eps(5e-3)
        {
        }

        CrossPolytopeArgs(std::istream& in) {
            in.read(reinterpret_cast<char*>(&estimation_repetitions), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&estimation_eps), sizeof(float));
        }

        void serialize(std::ostream& out) const {
            out.write(reinterpret_cast<const char*>(&estimation_repetitions), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&estimation_eps), sizeof(float));
        }

        void set_no_preprocessing() {
            estimation_repetitions = 0;
            estimation_eps = 2.0;
        }

        uint64_t memory_usage(DatasetDescription<UnitVectorFormat> dataset) const {
            return sizeof(CrossPolytopeHashFunction)
                + dataset.args*dataset.storage_len*sizeof(int16_t);
        }
    };

    /// An implementation of cross-polytope LSH using random rotations.
    /// 
    /// See ``FHTCrossPolytopeHash`` for a more efficient hash function using pseudo-random rotations instead.
    ///
    /// This LSH applies a random rotation to vectors and then maps each vector to the closest axis.
    /// This yields multiple bits per hash function. 
    /// Since there is no easy way to calculate collision probabilities,
    /// they are estimated via sampling instead.
    class CrossPolytopeHash {
    public:
        using Args = CrossPolytopeArgs;
        using Sim = CosineSimilarity;
        using Function = CrossPolytopeHashFunction;

    private:
        DatasetDescription<UnitVectorFormat> dataset;
        Args args;
        CrossPolytopeCollisionEstimates estimates;

    public:
        CrossPolytopeHash(
            DatasetDescription<UnitVectorFormat> dataset,
            Args args
        )
          : dataset(dataset),
            args(args),
            estimates(
                (1 << ceil_log(dataset.args)),
                args.estimation_repetitions,
                args.estimation_eps)
        {
        }

        CrossPolytopeHash(std::istream& in)
          : dataset(in),
            args(in),
            estimates(in)
        {
        }

        void serialize(std::ostream& out) const {
            dataset.serialize(out);
            args.serialize(out);
            estimates.serialize(out);
        }

        CrossPolytopeHashFunction sample() {
            return CrossPolytopeHashFunction(dataset);
        }

        unsigned int bits_per_function() {
            return ceil_log(dataset.args)+1;
        }

        float collision_probability(
            float similarity,
            int_fast8_t num_bits
        ) const {
            return estimates.get_collision_probability(similarity, num_bits);
        }
    };
}
