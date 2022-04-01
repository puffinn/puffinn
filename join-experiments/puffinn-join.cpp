#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include "protocol.hpp"
#include "puffinn.hpp"
#include "puffinn/performance.hpp"

const unsigned long long MB = 1024*1024;

int main(void) {
    std::string protocol_line;

    // Read parameters
    expect("setup");
    // std::cerr << "[c++] setup" << std::endl;
    unsigned int k = 10;
    float recall = 0.8;
    std::string method = "BF";
    unsigned long long space_usage = 100*MB;
    while (true) {
        std::getline(std::cin, protocol_line);
        // std::cerr << "[c++] setup line: " << protocol_line << std::endl;
        if (protocol_line == "sppv1 end") {
            break;
        }
        std::istringstream line(protocol_line);
        std::string key;
        line >> key;
        if (key == "k") {
            line >> k;
        } else if (key == "recall") {
            line >> recall;
        } else if (key == "method") {
            line >> method;
        } else if (key == "space_usage") {
            line >> space_usage;
            space_usage *= MB;
        } else {
            std::cout << "sppv1 err unknown parameter " << key << std::endl;
            return -1;
        }
    }
    send("ok");

    // Read the dataset
    expect("data");
    // std::cerr << "[c++] receiving data" << std::endl;
    auto dataset = read_vectors_stdin();
    auto dimensions = dataset[0].size(); 
    send("ok");

    expect("index");
    // Construct the search index.
    // Here we use the cosine similarity measure with the default hash functions.
    // The index expects vectors with the same dimensionality as the first row of the dataset
    // and will use at most the specified amount of memory.
    puffinn::Index<puffinn::CosineSimilarity, puffinn::SimHash> index(
        dimensions,
        space_usage,
        // puffinn::TensoredHashArgs<puffinn::SimHash>()
        puffinn::IndependentHashArgs<puffinn::SimHash>()
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

    expect("workload");
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
    // std::cerr << "[c++] results size " << res.size() << std::endl; 
    for (auto v : res) {
        for (auto i : v) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
    send("end");

    // auto total_time =  puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Total);
    // auto search_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Search);
    // auto filter_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Filtering);
    // auto init_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::SearchInit);
    // auto indexing_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Indexing);
    // auto rebuild_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Rebuilding);
    // auto sorting_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Sorting);
    // auto index_hashing_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::IndexHashing);
    // auto index_sketching_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::IndexSketching);

    // std::stringstream ss;

    // ss << "search_time=" <<  search_time
    //     << "; filter_time=" << filter_time
    //     << "; init_time=" << init_time 
    //     << "; total_time=" << total_time;

    // std::cout 
    //     << "indexing_time=" << indexing_time
    //     << "\n\tsketching_time=" << index_sketching_time
    //     << "\n\thashing_time=" << index_hashing_time
    //     << "\n\trebuilding_time=" << rebuild_time
    //     << "\n\tsorting_time" << sorting_time
    //     << "\nsearch_time=" <<  search_time
    //     << "\nfilter_time=" << filter_time
    //     << "\ninit_time=" << init_time 
    //     << "\ntotal_time=" << total_time << std::endl;

}
