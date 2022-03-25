#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <highfive/H5Attribute.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>
#include "puffinn.hpp"
#include "puffinn/performance.hpp"

template <typename DataFormat>
std::vector<DataFormat> load_dataset(char * hdf5_path);

template <>
std::vector<std::vector<uint32_t>> load_dataset(char * hdf5_path) {
    HighFive::File file(hdf5_path, HighFive::File::ReadOnly);
    std::vector<uint32_t> items = H5Easy::load<std::vector<uint32_t>>(file, "vectors/items");
    std::vector<uint32_t> offsets = H5Easy::load<std::vector<uint32_t>>(file, "vectors/offsets");
    std::vector<uint32_t> lengths = H5Easy::load<std::vector<uint32_t>>(file, "vectors/lengths");

    std::vector<std::vector<uint32_t>> vectors;
    for (size_t i = 0; i < offsets.size(); i++) {
        auto start = items.begin() + offsets[i];
        auto end = start + lengths[i];
        std::vector<uint32_t> v(start, end);
        vectors.push_back(v);
    }
    return vectors;
}

template <>
std::vector<std::vector<float>> load_dataset(char * hdf5_path) {
    HighFive::File file(hdf5_path, HighFive::File::ReadOnly);
    std::vector<std::vector<float>> vectors = H5Easy::load<std::vector<std::vector<float>>>(file, "vectors");
    return vectors;
}

std::string get_type(char * hdf5_path) {
    HighFive::File file(hdf5_path, HighFive::File::ReadOnly);
    std::string type;
    file.getAttribute("type").read(type);
    return type;
}

template<typename DataFormat>
size_t get_dimensions(char * hdf5_path);

template<>
size_t get_dimensions<std::vector<uint32_t>>(char * hdf5_path) {
    HighFive::File file(hdf5_path, HighFive::File::ReadOnly);
    size_t universe;
    file.getAttribute("universe").read(&universe);
    puffinn::Dataset<puffinn::SetFormat> dataset(universe);
    return universe;
}

template<>
size_t get_dimensions<std::vector<float>>(char * hdf5_path) {
    HighFive::File file(hdf5_path, HighFive::File::ReadOnly);
    auto dimensions_attr = file.getAttribute("dimensions");
    size_t dimensions;
    dimensions_attr.read(&dimensions);
    puffinn::Dataset<puffinn::UnitVectorFormat> dataset(dimensions);
    return dimensions;
}

void print_stats() {
    auto total_time =  puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Total);
    auto search_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Search);
    auto filter_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Filtering);
    auto init_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::SearchInit);
    auto indexing_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Indexing);
    auto rebuild_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Rebuilding);
    auto sorting_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Sorting);
    auto index_hashing_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::IndexHashing);
    auto index_sketching_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::IndexSketching);
    std::cerr
        << "indexing_time=" << indexing_time
        << "\n\tsketching_time=" << index_sketching_time
        << "\n\thashing_time=" << index_hashing_time
        << "\n\trebuilding_time=" << rebuild_time
        << "\n\tsorting_time" << sorting_time
        << "\nsearch_time=" <<  search_time
        << "\nfilter_time=" << filter_time
        << "\ninit_time=" << init_time 
        << "\ntotal_time=" << total_time << std::endl;
}

template <typename DataFormat, typename Similarity, typename HashFunction>
void run(char * filename, uint32_t k, float recall, std::string method, size_t space_usage) {
    auto start_time = std::chrono::steady_clock::now();
    auto dataset = load_dataset<DataFormat>(filename);
    size_t dimensions = get_dimensions<DataFormat>(filename);
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cerr << "Data loaded in " << elapsed << " ms" << std::endl;

    // Construct the search index.
    puffinn::Index<Similarity, HashFunction> index(
        dimensions,
        space_usage,
        puffinn::TensoredHashArgs<HashFunction>()
    );
    // Insert each vector into the index.
    for (auto v : dataset) {
        index.insert(v);
    }
    start_time = std::chrono::steady_clock::now();
    std::cerr << "Building the index. This can take a while..." << std::endl; 
    // Rebuild the index to include the inserted points
    index.rebuild(false);
    end_time = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    auto throughput = ((float) dataset.size()) / (elapsed / 1000.0);
    std::cerr << "Index built in " << elapsed << " ms " << throughput << " vecs/s" << std::endl;

    start_time = std::chrono::steady_clock::now();
    std::cerr << "Computing the join using " << method << ". This can take a while." << std::endl;    
    std::vector<std::vector<uint32_t>>  res;
    if (method == "BF") {
        res = index.bf_join(k);
    } else if (method == "BFGlobal") {
        index.global_bf_join(k);
    } else if (method == "LSH") {
        res = index.naive_lsh_join(k, recall);
    } else if (method == "LSHJoin") {
        res = index.lsh_join(k, recall);
    } else if (method == "LSHJoinGlobal") {
        index.global_lsh_join(k, recall);
    }
    end_time = std::chrono::steady_clock::now();
    auto elapsed_join = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    throughput = ((float) dataset.size()) / (elapsed_join / 1000.0);
    std::cerr << "Join computed in " << elapsed_join << " ms " << throughput << " queries/s" << std::endl;

    print_stats();

}

const unsigned long long MB = 1024*1024;

// Takes the following arguments: filename (num_neighbors) (recall) (space_usage in MB)
// The recall is a lower bound on the probability of finding each of the closest neighbors and is
// between 0 and 1.
int main(int argc, char* argv[]) {
    if (argc != 6) {
        printf("USAGE: PuffinnJoin filename k recall method space_usage");
        return 1;
    }
    // Read parameters
    char * filename = argv[1];
    uint32_t k = atoi(argv[2]);
    float recall = atof(argv[3]);
    std::string method = argv[4];
    size_t space_usage = atoi(argv[5])*MB;

    auto dataset_type = get_type(filename);

    if (dataset_type == "dense") {
        run<
            std::vector<float>,
            puffinn::CosineSimilarity, 
            puffinn::SimHash
        >(filename, k, recall, method, space_usage);
    } else if (dataset_type == "sparse") {
        run<
            std::vector<uint32_t>,
            puffinn::JaccardSimilarity,
            puffinn::MinHash1Bit
        >(filename, k, recall, method, space_usage);
    } else {
        printf("Unknown dataset type %s\n", dataset_type.c_str());
        return 1;
    }


    return 0;
}


