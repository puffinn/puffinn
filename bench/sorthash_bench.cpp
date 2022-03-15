#include "nanobench.h"
#include "sorthash.hpp"
#include "puffinn/hash/simhash.hpp"
#include "puffinn/hash/crosspolytope.hpp"
#include "puffinn/hash_source/pool.hpp"
#include "puffinn/hash_source/independent.hpp"
#include "puffinn/hash_source/tensor.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "puffinn/similarity_measure/jaccard.hpp"
#include "puffinn/dataset.hpp"
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>

puffinn::Dataset<puffinn::CosineSimilarity::Format> read_glove(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::invalid_argument("File not found");
    }

    std::vector<std::vector<float>> vectors;
    while (!file.eof()) {
        std::string full_line;
        std::getline(file, full_line);
        std::istringstream line(full_line);
        
        std::string word;
        line >> word;

        std::vector<float> row;
        float val;
        while (line >> val) {
            row.push_back(val);
        }

        if (row.size() != 0) {
            vectors.push_back(row);
        }
    }

    auto dimensions = vectors[0].size(); 
    puffinn::Dataset<puffinn::CosineSimilarity::Format> dataset(dimensions);
    for (auto v : vectors) {
        dataset.insert(v);
    }
    return dataset;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "USAGE: Bench <FILE>" << std::endl;
        return 1;
    }
    auto dataset = read_glove(argv[1]);

    size_t runs = 100;

    // Compute hashes
    auto hash_source = puffinn::IndependentHashArgs<puffinn::FHTCrossPolytopeHash>().build(
        dataset.get_description(),
        1,
        puffinn::MAX_HASHBITS);
    auto hash_function = hash_source->sample();
    std::vector<uint32_t> hashes;
    for (size_t i = 0; i < dataset.get_size(); i++) {
        auto state = hash_source->reset(dataset[i], true);
        hashes.push_back((*hash_function)(state.get()));
    }

    printf("# Sorting actual Glove hashes\n\n");
    printf("| algorithm               |          n |    time (ns) |    ns/elem |    throghput |\n");
    printf("| :---------------------- | ---------: | -----------: | ---------: | -----------: |\n");
    {
        uint64_t total_ns = 0;
        for (size_t run = 0; run < runs; run++) {
            std::vector<uint32_t> tosort(hashes);
            auto start = std::chrono::steady_clock::now();
            std::sort(tosort.begin(), tosort.end());
            auto end = std::chrono::steady_clock::now();
            total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        double avg_ns = total_ns / runs;
        double per_element = avg_ns / hashes.size();
        double throughput = ((double)hashes.size()) / (avg_ns / 1000000000.0);
        printf("| std::sort               | %10lu | %12.0f | %10.2f | %12.2f |\n", hashes.size(), avg_ns, per_element, throughput);
    }
    {
        uint64_t total_ns = 0;
        std::vector<uint32_t> aux;
        for (size_t run = 0; run < runs; run++) {
            std::vector<uint32_t> tosort(hashes);
            auto start = std::chrono::steady_clock::now();
            puffinn::sort_hashes_24(tosort, aux);
            auto end = std::chrono::steady_clock::now();
            total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        double avg_ns = total_ns / runs;
        double per_element = avg_ns / hashes.size();
        double throughput = ((double)hashes.size()) / (avg_ns / 1000000000.0);
        printf("| puffinn::sort_hashes_24 | %10lu | %12.0f | %10.2f | %12.2f |\n", hashes.size(), avg_ns, per_element, throughput);
    }



    // Benchmark uniform random numbers
    std::mt19937 generator (1234);
    std::uniform_int_distribution<uint32_t> distribution(0,1 << 23);

    std::vector<uint32_t> aux;
    std::vector<uint32_t> numbers;

    printf("\n\n# Sorting uniformly distributed data\n\n");
    printf("| algorithm               |          n |    time (ns) |    ns/elem |    throghput |\n");
    printf("| :---------------------- | ---------: | -----------: | ---------: | -----------: |\n");
    size_t ns[] = {1000, 10000, 100000, 1000000, 10000000};
    for (const size_t n : ns) {
        numbers.clear();
        for (size_t i = 0; i < n; i++) {
            numbers.push_back(distribution(generator));
        }
        
        // std::sort
        {
            uint64_t total_ns = 0;
            for (size_t run = 0; run < runs; run++) {
                std::vector<uint32_t> tosort(numbers);
                auto start = std::chrono::steady_clock::now();
                std::sort(tosort.begin(), tosort.end());
                auto end = std::chrono::steady_clock::now();
                total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            }
            double avg_ns = total_ns / runs;
            double per_element = avg_ns / numbers.size();
            double throughput = ((double)numbers.size()) / (avg_ns / 1000000000.0);
            printf("| std::sort               | %10lu | %12.0f | %10.2f | %12.2f |\n", numbers.size(), avg_ns, per_element, throughput);
        }

        // puffinn::sort_hashes_24
        {
            uint64_t total_ns = 0;
            for (size_t run = 0; run < runs; run++) {
                std::vector<uint32_t> tosort(numbers);
                auto start = std::chrono::steady_clock::now();
                puffinn::sort_hashes_24(tosort, aux);
                auto end = std::chrono::steady_clock::now();
                total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            }
            double avg_ns = total_ns / runs;
            double per_element = avg_ns / numbers.size();
            double throughput = ((double)numbers.size()) / (avg_ns / 1000000000.0);
            printf("| puffinn::sort_hashes_24 | %10lu | %12.0f | %10.2f | %12.2f |\n", numbers.size(), avg_ns, per_element, throughput);
        }
    }
}
