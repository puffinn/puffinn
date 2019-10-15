#pragma once

#if defined(__AVX2__) || defined(__AVX__)
    #include <immintrin.h>
#endif



namespace puffinn {
    #ifdef __AVX2__
        static int16_t dot_product_i16_avx2(const int16_t* lhs, const int16_t* rhs, unsigned int dimensions) {
            // Number of i16 values that fit into a 256 bit vector.
            const static unsigned int VALUES_PER_VEC = 16;

            // specialized function for multiplication of the fixed point format
            __m256i res = _mm256_mulhrs_epi16(
                _mm256_load_si256((__m256i*)&lhs[0]),
                _mm256_load_si256((__m256i*)&rhs[0]));
            for (
                unsigned int i=VALUES_PER_VEC;
                i < dimensions;
                i += VALUES_PER_VEC
            ) {
                __m256i tmp = _mm256_mulhrs_epi16(
                    _mm256_load_si256((__m256i*)&lhs[i]),
                    _mm256_load_si256((__m256i*)&rhs[i]));
                res = _mm256_add_epi16(res, tmp);
            }
            alignas(32) int16_t stored[VALUES_PER_VEC];
            _mm256_store_si256((__m256i*)stored, res);
            int16_t ret = 0;
            for (unsigned i=0; i<VALUES_PER_VEC; i++) { ret += stored[i]; }
            return ret;
        }
    #endif

    static int16_t dot_product_i16_simple(const int16_t* lhs, const int16_t* rhs, unsigned int dimensions) {
        int16_t res = 0.0;
        for (unsigned int i=0; i < dimensions; i++) {
            res += lhs[i]*rhs[i];
        }
        return res;
    }

    static int16_t dot_product_i16(const int16_t* lhs, const int16_t* rhs, unsigned int dimensions) {
        #ifdef __AVX2__
            return dot_product_i16_avx2(lhs, rhs, dimensions);
        #else
            return dot_product_i16_simple(lhs, rhs, dimensions);
        #endif
    }

    #ifdef __AVX__
        // Compute the l2 distance between two floating point vectors without taking the
        // final root.
        static float l2_distance_float_avx(const float* lhs, const float* rhs, unsigned int dimensions) {
            // Number of float values that fit into a 256 bit vector.
            const static unsigned int VALUES_PER_VEC = 8;

            __m256 res = _mm256_sub_ps(
                _mm256_load_ps(&lhs[0]),
                _mm256_load_ps(&rhs[0]));
            res = _mm256_mul_ps(res, res);

            for (
                unsigned int i=VALUES_PER_VEC;
                i < dimensions;
                i += VALUES_PER_VEC
            ) {
                __m256 tmp = _mm256_sub_ps(
                    _mm256_load_ps(&lhs[i]),
                    _mm256_load_ps(&rhs[i]));
                tmp = _mm256_mul_ps(tmp, tmp);
                res = _mm256_add_ps(res, tmp);
            }
            alignas(32) float stored[VALUES_PER_VEC];
            _mm256_store_ps(stored, res);
            float ret = 0;
            for (unsigned i=0; i < VALUES_PER_VEC; i++) {
                ret += stored[i];
            }
            return ret;
        }
    #endif

    static float l2_distance_float_simple(const float* lhs, const float* rhs, unsigned int dimensions) {
        float res = 0.0;
        for (unsigned int i=0; i < dimensions; i++) {
            float diff = lhs[i]-rhs[i];
            res += diff*diff;
        }
        return res;
    }

    static float l2_distance_float(const float* lhs, const float* rhs, unsigned int dimensions) {
        #ifdef __AVX__
            return l2_distance_float_avx(lhs, rhs, dimensions);
        #else
            return l2_distance_float_simple(lhs, rhs, dimensions);
        #endif
    }

    // Round up to nearest power of two.
    constexpr static unsigned int ceil_log(unsigned int value) {
        unsigned int log = 0;
        unsigned int power_of_two = 1;
        while (power_of_two < value) {
            log++;
            power_of_two *= 2;
        }
        return log;
    }
}
