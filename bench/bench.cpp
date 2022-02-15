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

void bench_index_build(const std::vector<std::vector<float>> & dataset) {
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

void bench_query(const std::vector<std::vector<float>> & dataset) {
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

void bench_hash(const std::vector<std::vector<float>> & vectors) {
    auto dimensions = vectors[0].size(); 

    puffinn::Dataset<puffinn::CosineSimilarity::Format> dataset(dimensions);
    for (auto v : vectors) {
        dataset.insert(v);
    }

    // To benchmark the index build time we have to start from a new
    // index at each measurement iteration, otherwise the index is already populated
    // and no rebuild is triggered.
    auto bencher = ankerl::nanobench::Bench()
        .title("Hashing")
        .minEpochIterations(100)
        .timeUnit(std::chrono::nanoseconds(1), "ns");
    
    // auto desc = dataset.get_description();
    // auto stored_v = to_stored_type<puffinn::CosineSimilarity::Format>(dataset[0], desc);
    auto vec = dataset[0];

    puffinn::FHTCrossPolytopeHash fhtcp(dataset.get_description(), puffinn::FHTCrossPolytopeArgs());
    auto hash_fhtcp = fhtcp.sample();
    bencher.run("FHT cross polytope", [&] {
        ankerl::nanobench::doNotOptimizeAway(hash_fhtcp(vec));
    });

    puffinn::SimHash simhash(dataset.get_description(), puffinn::SimHashArgs());
    auto hash_simhash = simhash.sample();
    bencher.run("SimHash", [&] {
        ankerl::nanobench::doNotOptimizeAway(hash_simhash(vec));
    });
}


int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "USAGE: Bench <FILE>" << std::endl;
        return 1;
    }
    auto dataset = read_glove(argv[1]);

    bench_query(dataset);
    // bench_index_build(dataset);
    bench_hash(dataset);
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