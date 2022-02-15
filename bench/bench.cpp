#include <nanobench.h>

#include "puffinn/collection.hpp"
#include "puffinn/hash/simhash.hpp"
#include "puffinn/hash/crosspolytope.hpp"
#include "puffinn/hash_source/pool.hpp"
#include "puffinn/hash_source/independent.hpp"
#include "puffinn/hash_source/tensor.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "puffinn/similarity_measure/jaccard.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

const unsigned int MB = 1024*1024;

std::vector<std::vector<float>> read_glove(const std::string& filename);

void bench_index_build(std::vector<std::vector<float>> dataset) {
    auto dimensions = dataset[0].size(); 

    // To benchmark the index build time we have to start from a new
    // index at each measurement iteration, otherwise the index is already populated
    // and no rebuild is triggered.
    auto bencher = ankerl::nanobench::Bench()
        .title("Index building")
        .timeUnit(std::chrono::milliseconds(1), "ms");
    auto index_memory = 100*MB;

    // Therefore, first we bench how much time it takes to push things into the index
    bencher.run("index_insert_data", [&] {
        puffinn::Index<puffinn::CosineSimilarity> index(
            dimensions,
            index_memory
        );
        for (auto v : dataset) { index.insert(v); }
    });
    
    // Therefore, first we bench how much time it takes to push things into the index
    bencher.run("index_rebuild", [&] {
        puffinn::Index<puffinn::CosineSimilarity> index(
            dimensions,
            index_memory
        );
        for (auto v : dataset) { index.insert(v); }
        index.rebuild();
    });
}

void bench_query(std::vector<std::vector<float>> dataset) {
    auto dimensions = dataset[0].size(); 

    // To benchmark the index build time we have to start from a new
    // index at each measurement iteration, otherwise the index is already populated
    // and no rebuild is triggered.
    auto bencher = ankerl::nanobench::Bench()
        .title("Index query")
        .minEpochIterations(100)
        .timeUnit(std::chrono::nanoseconds(1), "ns");
    auto index_memory = 100*MB;

    puffinn::Index<puffinn::CosineSimilarity> index(
        dimensions,
        index_memory
    );
    for (auto v : dataset) { index.insert(v); }
    index.rebuild();
    
    bencher.run("index_query (query 0)", [&] {
        index.search(dataset[0], 11, 0.9);
    });
    bencher.run("index_query (query 100)", [&] {
        index.search(dataset[100], 11, 0.9);
    });
    bencher.run("index_query (query 1000)", [&] {
        index.search(dataset[1000], 11, 0.9);
    });
}


int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "USAGE: Bench <FILE>" << std::endl;
        return 1;
    }
    auto dataset = read_glove(argv[1]);

    bench_query(dataset);
    bench_index_build(dataset);
}

std::vector<std::vector<float>> read_glove(const std::string& filename) {
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

    return vectors;
}