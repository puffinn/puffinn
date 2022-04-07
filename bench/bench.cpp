#include <nanobench.h>

#include "puffinn/collection.hpp"
#include "puffinn/hash/simhash.hpp"
#include "puffinn/hash/crosspolytope.hpp"
#include "puffinn/hash_source/pool.hpp"
#include "puffinn/hash_source/independent.hpp"
#include "puffinn/hash_source/tensor.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "puffinn/similarity_measure/jaccard.hpp"
#include "alternatives.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <omp.h>

const unsigned int MB = 1024*1024;

std::vector<std::vector<float>> read_glove(const std::string& filename);

void bench_api_simhash(
    const std::vector<std::vector<float>> & dataset
) {
    auto bencher = ankerl::nanobench::Bench()
        .title("Simhash computations")
        .minEpochIterations(100)
        .batch(dataset.size())
        .timeUnit(std::chrono::nanoseconds(1), "ns");

    auto dimensions = dataset[0].size(); 
    puffinn::Dataset<puffinn::UnitVectorFormat> dat(dimensions);
    for (auto v : dataset) { dat.insert(v); }
    size_t n = dataset.size();

    auto source = puffinn::IndependentHashArgs<puffinn::SimHash>().build(
        dat.get_description(), 1, 24
    );

    auto hash_fn = source->sample();

    bencher.run("old API", [&] {
        for (size_t i=0; i<n; i++) {
            auto state = source->reset(dat[i], false);
            ankerl::nanobench::doNotOptimizeAway((*hash_fn)(state.get()));
        }
    });

    std::vector<uint32_t> hashes;
    bencher.run("new API", [&] {
        for (size_t i=0; i<n; i++) {
            source->hash_repetitions(dat[i], hashes);
        }
    });
}

template<typename THash, typename THashSourceArgs>
void do_build_index(ankerl::nanobench::Bench * bencher, const char * name, const std::vector<std::vector<float>> & dataset, double index_memory) {
    auto dimensions = dataset[0].size(); 
    bencher->run(name, [&] {
        puffinn::Index<puffinn::CosineSimilarity, THash> index(
            dimensions,
            index_memory,
            THashSourceArgs()
        );
        for (auto v : dataset) { index.insert(v); }
        index.rebuild(false);
    });
}

void bench_index_build(const std::vector<std::vector<float>> & dataset) {
    printf("Benchmarking index build\n\n");
    auto dimensions = dataset[0].size(); 

    // To benchmark the index build time we have to start from a new
    // index at each measurement iteration, otherwise the index is already populated
    // and no rebuild is triggered.
    auto bencher = ankerl::nanobench::Bench();
    // auto index_memory = 100*MB;
    
    // memory is set so that we have 600 tables
    bencher.title("Simhash independent");
    omp_set_num_threads(1);
    do_build_index<puffinn::SimHash, puffinn::IndependentHashArgs<puffinn::SimHash>>(&bencher, "1 thread", dataset, 537*MB); // 74 MB
    omp_set_num_threads(2);
    do_build_index<puffinn::SimHash, puffinn::IndependentHashArgs<puffinn::SimHash>>(&bencher, "2 threads", dataset, 537*MB); // 74 MB
    omp_set_num_threads(4);
    do_build_index<puffinn::SimHash, puffinn::IndependentHashArgs<puffinn::SimHash>>(&bencher, "4 threads", dataset, 537*MB); // 74 MB
    omp_set_num_threads(8);
    do_build_index<puffinn::SimHash, puffinn::IndependentHashArgs<puffinn::SimHash>>(&bencher, "8 threads", dataset, 537*MB); // 74 MB
    omp_set_num_threads(16);
    do_build_index<puffinn::SimHash, puffinn::IndependentHashArgs<puffinn::SimHash>>(&bencher, "16 threads", dataset, 537*MB); // 74 MB
    omp_set_num_threads(32);
    do_build_index<puffinn::SimHash, puffinn::IndependentHashArgs<puffinn::SimHash>>(&bencher, "32 threads", dataset, 537*MB); // 74 MB

    bencher.title("Simhash tensored");
    omp_set_num_threads(1);
    do_build_index<puffinn::SimHash, puffinn::TensoredHashArgs<puffinn::SimHash>>(&bencher, "1 thread", dataset, 537*MB); // 74 MB
    omp_set_num_threads(2);
    do_build_index<puffinn::SimHash, puffinn::TensoredHashArgs<puffinn::SimHash>>(&bencher, "2 threads", dataset, 537*MB); // 74 MB
    omp_set_num_threads(4);
    do_build_index<puffinn::SimHash, puffinn::TensoredHashArgs<puffinn::SimHash>>(&bencher, "4 threads", dataset, 537*MB); // 74 MB
    omp_set_num_threads(8);
    do_build_index<puffinn::SimHash, puffinn::TensoredHashArgs<puffinn::SimHash>>(&bencher, "8 threads", dataset, 537*MB); // 74 MB
    omp_set_num_threads(16);
    do_build_index<puffinn::SimHash, puffinn::TensoredHashArgs<puffinn::SimHash>>(&bencher, "16 threads", dataset, 537*MB); // 74 MB
    omp_set_num_threads(32);
    do_build_index<puffinn::SimHash, puffinn::TensoredHashArgs<puffinn::SimHash>>(&bencher, "32 threads", dataset, 537*MB); // 74 MB

    // do_build_index<puffinn::SimHash, puffinn::TensoredHashArgs<puffinn::SimHash>>(&bencher, "SimHash tensored", dataset, 534*MB); // 70.6 MB
    // do_build_index<puffinn::FHTCrossPolytopeHash, puffinn::IndependentHashArgs<puffinn::FHTCrossPolytopeHash>>(&bencher, "FHT CrossPolytope independent", dataset, 534.5*MB); // 71.2 MB
    // do_build_index<puffinn::FHTCrossPolytopeHash, puffinn::TensoredHashArgs<puffinn::FHTCrossPolytopeHash>>(&bencher, "FHT CrossPolytope tensored", dataset, 534*MB); // 70.5 MB
}


void bench_join(const std::vector<std::vector<float>> & dataset) {
    auto dimensions = dataset[0].size(); 
    auto index_memory = 100*MB;

    puffinn::Index<puffinn::CosineSimilarity> index(
        dimensions,
        index_memory
    );

    for (auto v : dataset) { index.insert(v); }
    index.rebuild(false);

    auto bencher = ankerl::nanobench::Bench()
        .title("LSH join")
        .timeUnit(std::chrono::milliseconds(1), "ms");

    bencher.run("LSH join", [&] {
        index.lsh_join(10, 0.9);
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
        puffinn::MAX_HASHBITS);
    auto hash_function = hash_source->sample();
    bencher->run(name, [&] {
        auto state = hash_source->reset(dataset[0], true);
        ankerl::nanobench::doNotOptimizeAway((*hash_function)(state.get()));
    });
}

template<typename THash>
void run_static(ankerl::nanobench::Bench * bencher, const char * name, const puffinn::Dataset<puffinn::UnitVectorFormat> & dataset) {
    puffinn::IndependentHasherStatic<THash> hash_fun(
        dataset.get_description(),
        typename THash::Args(),
        1,
        puffinn::MAX_HASHBITS
    );

    auto v = dataset[0];
    bencher->run(name, [&] {
        ankerl::nanobench::doNotOptimizeAway(hash_fun.hash(0, v));
    });
}

template<typename THash> 
void run_single_hash(ankerl::nanobench::Bench * bencher, const char * name, const puffinn::Dataset<puffinn::UnitVectorFormat> & dataset) {
    THash hash(dataset.get_description(), typename THash::Args());
    auto hash_fn = hash.sample();
    bencher->run(name, [&] {
        ankerl::nanobench::doNotOptimizeAway(hash_fn(dataset[0]));
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
        .minEpochIterations(1000)
        .timeUnit(std::chrono::nanoseconds(1), "ns");
    
    run_single_hash<puffinn::CrossPolytopeHash>(&bencher, "Cross polytope (single)", dataset);
    run_with_indirection<puffinn::CrossPolytopeHash>(&bencher, "Cross polytope (indirection)", dataset);
    run_static<puffinn::CrossPolytopeHash>(&bencher, "Cross polytope (static)", dataset);

    run_single_hash<puffinn::FHTCrossPolytopeHash>(&bencher, "FHT Cross polytope (single)", dataset);
    run_with_indirection<puffinn::FHTCrossPolytopeHash>(&bencher, "FHT cross polytope (indirection)", dataset);
    run_static<puffinn::FHTCrossPolytopeHash>(&bencher, "FHT cross polytope (static)", dataset);

    run_single_hash<puffinn::SimHash>(&bencher, "SimHash (single)", dataset);
    run_with_indirection<puffinn::SimHash>(&bencher, "SimHash (indirection)", dataset);
    run_static<puffinn::SimHash>(&bencher, "SimHash (static)", dataset);
}


int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "USAGE: Bench <FILE>" << std::endl;
        return 1;
    }
    auto dataset = read_glove(argv[1]);

    // bench_api_simhash(dataset);
    // bench_query(dataset);
    // bench_index_build(dataset);
    // bench_hash(dataset);
    bench_join(dataset);
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