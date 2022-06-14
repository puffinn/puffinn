#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <omp.h>
#include "protocol.hpp"
#include "puffinn.hpp"
#include "puffinn/performance.hpp"

const unsigned long long MB = 1024*1024;

template<typename RawData> 
std::pair<std::vector<RawData>, size_t> do_read_vectors();

template<> 
std::pair<std::vector<std::vector<float>>, size_t> do_read_vectors<std::vector<float>>() {
    auto data = read_float_vectors_hdf5(true);
    std::cerr << "loaded dataset with " << data.size() << " points of dimension " << data[0].size() << std::endl;
    return { data, data[0].size() };
}

template<> 
std::pair<std::vector<std::vector<uint32_t>>, size_t> do_read_vectors<std::vector<uint32_t>>() {
    auto data = read_int_vectors_hdf5();
    size_t universe = 0;
    for (auto & v : data) {
      for (auto x : v) {
        if (x > universe) {
          universe = x;
        }
      }
    }
    universe++;
    return { data, universe };
}

template<typename Similarity, typename HashSourceArgs, typename HashFn, typename RawData>
void run_index(std::vector<RawData> dataset, size_t dimensions, size_t space_usage) {
    // Construct the search index.
    // Here we use the cosine similarity measure with the default hash functions.
    // The index expects vectors with the same dimensionality as the first row of the dataset
    // and will use at most the specified amount of memory.
    puffinn::Index<Similarity, HashFn> index(
        dimensions,
        space_usage,
        // puffinn::TensoredHashArgs<puffinn::SimHash>()
        HashSourceArgs()
    );
    // Insert each vector into the index.
    for (auto v : dataset) { index.insert(v); }
    auto start_time = std::chrono::steady_clock::now();
    std::cerr << "Building the index. This can take a while..." << std::endl; 
    // Rebuild the index to include the inserted points
    index.rebuild(false);
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = (end_time - start_time);
    auto throughput = ((float) dataset.size()) / elapsed.count();
    std::cerr << "Index built in " << elapsed.count() << " s " << throughput << " vecs/s" << std::endl;
    send("ok");

    // now we accept multiple workloads on the same index, until we receive "end_workloads"
    while(true) {
        std::string next_workload = protocol_read();
        std::cerr << "received " << next_workload << std::endl;
        if (next_workload == "end_workloads") {
            break;
        }
        std::string workload_params_str = next_workload.substr(std::string("workload ").size());
        std::cerr << "NEW WORKLOAD ON INDEX " << workload_params_str << std::endl;

        // query params
        unsigned int k = 10;
        float recall = 0.8;
        std::string method = "BF";
        float brute_force_perc = 0.0;

        std::istringstream workload_params_stream(workload_params_str);
        while (true) {
            std::string key;
            workload_params_stream >> key;
            if (key == "") {
                break;
            }
            if (key == "k") {
                workload_params_stream >> k;
            } else if (key == "recall") {
                workload_params_stream >> recall;
            } else if (key == "method") {
                workload_params_stream >> method;
            } else if (key == "brute_force_perc") {
                workload_params_stream >> brute_force_perc;
            } else {
                std::cout << "sppv1 err unknown parameter " << key << std::endl;
                throw std::invalid_argument("unknown parameter");
            }
        }

        start_time = std::chrono::steady_clock::now();
        std::vector<std::vector<uint32_t>>  res;
        if (method == "BF") {
            res = index.bf_join(k);
        } else if (method == "BFGlobal") {
            index.global_bf_join(k);
        } else if (method == "LSH") {
            res = index.naive_lsh_join(k, recall);
        } else if (method == "LSHJoin") {
            res = index.lsh_join(k, recall, brute_force_perc);
        } else if (method == "LSHJoinGlobal") {
            auto pairs = index.global_lsh_join(k, recall);
            for (auto entry : pairs.best_indices()) {
                std::vector<uint32_t> vpair;
                vpair.push_back(entry.first);
                vpair.push_back(entry.second);
                res.push_back(vpair);
            }
        }
        end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_join = (end_time - start_time);
        throughput = ((float) dataset.size()) / elapsed_join.count();
        std::cerr << "Join computed in " << elapsed_join.count() << " s " << throughput << " queries/s" << std::endl;
        send("ok");

        expect("result");
        for (auto v : res) {
            for (auto i : v) {
                std::cout << i << " ";
            }
            std::cout << std::endl;
        }
        std::cerr << "sent all items of result" << std::endl;
        send("end");

    }
}


template<typename Similarity, typename HashFn, typename RawData>
void run() {
    auto data_pair = do_read_vectors<RawData>();
    auto dataset = data_pair.first;
    auto dimensions = data_pair.second;
    send("ok");

    // index params
    std::string hash_source = "Independent"; // in alternative, Tensored
    unsigned long long space_usage = 100*MB;
    int threads = -1;

    std::string index_params_str = expect("index");
    std::cerr << "reading parameters from `" << index_params_str << "`" << std::endl;
    std::istringstream index_params_stream(index_params_str);

    while (true) {
        std::string key;
        index_params_stream >> key;
        std::cerr << "read key `" << key << "`" << std::endl;
        if (key == "") {
            break;
        }
        if (key == "hash_source") {
            index_params_stream >> hash_source;
        } else if (key == "space_usage") {
            index_params_stream >> space_usage;
            space_usage *= MB;
        } else if (key == "threads") {
            index_params_stream >> threads;
        } else {
            std::cout << "sppv1 err unknown parameter `" << key << "`" << std::endl;
            std::cerr << "sppv1 err unknown parameter `" << key << "`" << std::endl;
            throw std::invalid_argument("unknown parameter " + key);
        }
    }


    if (hash_source == "Independent") {
        run_index<Similarity, puffinn::IndependentHashArgs<HashFn>, HashFn, RawData>(
            dataset, dimensions, space_usage
        );
    } else if (hash_source == "Tensored") {
        run_index<Similarity, puffinn::TensoredHashArgs<HashFn>, HashFn, RawData>(
            dataset, dimensions, space_usage
        );
    } 
}

int main(void) {
    std::string protocol_line;

    // Read the dataset
    expect("data");
    std::string distance_type = protocol_read();
    std::cerr << "[c++] distance type "  << distance_type << std::endl;
    // we send the ack within the `run` function

    if (distance_type == "cosine" || distance_type == "angular") {
        run<puffinn::CosineSimilarity, puffinn::SimHash, std::vector<float>>();
    } else if (distance_type == "jaccard") {
        run<puffinn::JaccardSimilarity, puffinn::MinHash1Bit, std::vector<uint32_t>>();
    }
    std::cerr << "[c++] done" << std::endl;
}
