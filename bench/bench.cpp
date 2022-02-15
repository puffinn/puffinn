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

template<typename THash>
void run_with_indirection(ankerl::nanobench::Bench * bencher, const char * name, const puffinn::Dataset<puffinn::UnitVectorFormat> & dataset) {
    auto hash_args = puffinn::IndependentHashArgs<THash>();
    auto hash_source = hash_args.build(
        dataset.get_description(),
        1,
        1);
    auto hash_function = hash_source->sample();
    auto state = hash_source->reset(dataset[0], true);
    bencher->run(name, [&] {
        ankerl::nanobench::doNotOptimizeAway((*hash_function)(state.get()));
    });
}

template<typename THash> 
void run_no_indirection(ankerl::nanobench::Bench * bencher, const char * name, const puffinn::Dataset<puffinn::UnitVectorFormat> & dataset) {
    auto vec = dataset[0];
    THash hash(dataset.get_description(), typename THash::Args());
    auto hash_fn = hash.sample();
    bencher->run(name, [&] {
        ankerl::nanobench::doNotOptimizeAway(hash_fn(vec));
    });
}

void bench_hash(const std::vector<std::vector<float>> & vectors) {
    auto dimensions = vectors[0].size(); 

    puffinn::Dataset<puffinn::CosineSimilarity::Format> dataset(dimensions);
    for (auto v : vectors) {
        dataset.insert(v);
    }

    auto bencher = ankerl::nanobench::Bench()
        .title("Hashing")
        .minEpochIterations(100)
        .timeUnit(std::chrono::nanoseconds(1), "ns");
    
    run_no_indirection<puffinn::FHTCrossPolytopeHash>(&bencher, "FHT cross polytope", dataset);
    run_with_indirection<puffinn::FHTCrossPolytopeHash>(&bencher, "FHT cross polytope (indirection)", dataset);
    run_no_indirection<puffinn::SimHash>(&bencher, "SimHash", dataset);
    run_with_indirection<puffinn::SimHash>(&bencher, "SimHash (indirection)", dataset);
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