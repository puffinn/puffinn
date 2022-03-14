#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
#include <highfive/H5Attribute.hpp>
#include <highfive/H5File.hpp>
#include "puffinn.hpp"
#include "puffinn/performance.hpp"

struct Dataset {
    std::vector<std::string> words;
    std::map<std::string, std::vector<float>> vectors;

    static Dataset read_glove(const std::string& filename);
};

void write_result(
    const std::string& method_name,
    const std::string& ds_name,
    const std::vector<std::vector<uint32_t>>& res,
    const uint32_t num_tables,
    const float recall,
    const uint32_t k,
    const double time,
    const std::string& details
    ) {

    using namespace HighFive;

    try {

        std::stringstream ss;
        ss << ds_name << "/" << k << "/" << method_name;

        std::filesystem::create_directories(ss.str());
        
        ss << "/" << recall << "_" << num_tables << ".hdf5";
        File file(ss.str(), File::ReadWrite | File::Create | File::Truncate);

        DataSet results = file.createDataSet<uint32_t>("results", DataSpace::From(res));
        results.write(res);
        Attribute a = file.createAttribute<uint32_t>("k", DataSpace::From(k));
        a.write(k);
        a = file.createAttribute<float>("recall", DataSpace::From(recall));
        a.write(recall);
        a = file.createAttribute<uint32_t>("num_tables", DataSpace::From(num_tables));
        a.write(num_tables);
        a = file.createAttribute<double>("time", DataSpace::From(time));
        a.write(time);
        a = file.createAttribute<std::string>("details", DataSpace::From(details));
        a.write(details);

    } catch (Exception& err) {
        std::cerr << err.what() << std::endl;
    }
}



const unsigned long long GB = 1024*1024*1024;
const unsigned long long MB = 1024*1024;

// Takes the following arguments: filename (num_neighbors) (recall) (space_usage in MB)
// The recall is a lower bound on the probability of finding each of the closest neighbors and is
// between 0 and 1.
int main(int argc, char* argv[]) {
    // Read parameters
    std::string filename;
    unsigned int k = 10;
    float recall = 0.8;
    std::string method = "BF";
    unsigned long long space_usage = 100*MB;
    switch (argc) {
        case 6: space_usage = static_cast<unsigned long long>(std::atof(argv[5])); 
        case 5: method = std::string(argv[4]);
        case 4: recall = std::atof(argv[3]); 
        case 3: k = std::atoi(argv[2]);
        case 2: filename = argv[1];
                break;
        default:
            std::cerr << "Usage: " << argv[0]
                << " filename (number of neighbors) (recall) (BF|LSH|LSHJoin|LSHJoinGlobal) (number_of_tables)" << std::endl;
            return -1;
    }

    // Read the dataset
    std::cerr << "Reading the dataset..." << std::endl;
    auto dataset = Dataset::read_glove(filename);
    if (dataset.words.size() == 0) {
        std::cerr << "Empty dataset" << std::endl;
        return -2;
    }
    auto dimensions = dataset.vectors[dataset.words[0]].size(); 

    // Construct the search index.
    // Here we use the cosine similarity measure with the default hash functions.
    // The index expects vectors with the same dimensionality as the first row of the dataset
    // and will use at most the specified amount of memory.
    puffinn::Index<puffinn::CosineSimilarity, puffinn::SimHash> index(
        dimensions,
        space_usage,
        puffinn::TensoredHashArgs<puffinn::SimHash>()
        // puffinn::IndependentHashArgs<puffinn::SimHash>()
    );
    // Insert each vector into the index.
    for (auto word : dataset.words) { index.insert(dataset.vectors[word]); }
    auto start_time = std::chrono::steady_clock::now();
    std::cerr << "Building the index. This can take a while..." << std::endl; 
    // Rebuild the index to include the inserted points
    index.rebuild();
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = (end_time - start_time);
    auto throughput = ((float) dataset.words.size()) / elapsed.count();
    std::cerr << "Index built in " << elapsed.count() << " s " << throughput << " vecs/s" << std::endl;

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
    std::chrono::duration<double> elapsed_join = (end_time - start_time);
    throughput = ((float) dataset.words.size()) / elapsed_join.count();
    std::cerr << "Join computed in " << elapsed_join.count() << " s " << throughput << " queries/s" << std::endl;

    std::string dataset_fn(filename);
    auto slash_pos = dataset_fn.find_last_of("/");
    auto suffix_pos = dataset_fn.find_last_of(".");

    auto total_time =  puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Total);
    auto search_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Search);
    auto filter_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Filtering);
    auto init_time = puffinn::g_performance_metrics.get_total_time(puffinn::Computation::SearchInit);

    std::stringstream ss;

    ss << "search_time=" <<  search_time
        << "; filter_time=" << filter_time
        << "; init_time=" << init_time 
        << "; total_time=" << total_time;

    std::cout << "search_time:" <<  search_time
        << "\nfilter_time=" << filter_time
        << "\ninit_time=" << init_time 
        << "\ntotal_time=" << total_time << std::endl;

    write_result(method, 
        dataset_fn.substr(slash_pos + 1, suffix_pos - slash_pos - 1), 
        res, 
        index.get_repetitions(), 
        recall, 
        k, 
        elapsed.count() + elapsed_join.count(),
        ss.str());

}

// Read a vector collection in the format used by GloVe.
// Each line contains a word and a space-separated list of numbers. 
Dataset Dataset::read_glove(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::invalid_argument("File not found");
    }

    std::vector<std::string> words;
    std::map<std::string, std::vector<float>> vectors;
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
            words.push_back(word);
            vectors[word] = row;
        }
    }

    Dataset res;
    res.words = words;
    res.vectors = vectors;
    return res;
}

