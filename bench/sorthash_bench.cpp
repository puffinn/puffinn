#include "nanobench.h"
#include "sorthash.hpp"
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>

int main(int argc, char ** argv) {
    size_t runs = 10;

    std::mt19937 generator (1234);
    std::uniform_int_distribution<uint32_t> distribution(0,1 << 23);

    std::vector<uint32_t> aux;
    std::vector<uint32_t> numbers;

    printf("| algorithm               |          n |    time (ns) |    ns/elem |\n");
    printf("| :---------------------- | ---------: | -----------: | ---------: |\n");
    size_t ns[] = {1000, 10000, 100000, 1000000, 10000000, 100000000};
    for (const size_t n : ns) {
        numbers.clear();
        for (size_t i = 0; i < n; i++) {
            numbers.push_back(distribution(generator));
        }
        
        // std::sort
        {
            std::vector<uint32_t> tosort(numbers);
            uint64_t total_ns = 0;
            for (size_t run = 0; run < runs; run++) {
                auto start = std::chrono::steady_clock::now();
                std::sort(tosort.begin(), tosort.end());
                auto end = std::chrono::steady_clock::now();
                total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            }
            uint64_t avg_ns = total_ns / runs;
            double per_element = avg_ns / tosort.size();
            printf("| std::sort               | %10lu | %12llu | %10.2f |\n", tosort.size(), avg_ns, per_element);
        }

        // puffinn::sort_hashes_24
        {
            std::vector<uint32_t> tosort(numbers);
            uint64_t total_ns = 0;
            for (size_t run = 0; run < runs; run++) {
                auto start = std::chrono::steady_clock::now();
                puffinn::sort_hashes_24(tosort, aux);
                auto end = std::chrono::steady_clock::now();
                total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            }
            uint64_t avg_ns = total_ns / runs;
            double per_element = avg_ns / tosort.size();
            printf("| puffinn::sort_hashes_24 | %10lu | %12llu | %10.2f |\n", tosort.size(), avg_ns, per_element);
        }
    }
}
