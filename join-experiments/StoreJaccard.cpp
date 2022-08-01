/// Computes and stores the top-1000 Jaccard similarities for each vector in the given HDF5 file

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "highfive/H5Easy.hpp"

template<typename Iter>
float jaccard(Iter abegin, Iter aend, Iter bbegin, Iter bend) {
    size_t intersection = 0;
    size_t n_a = aend - abegin;
    size_t n_b = bend - bbegin;
    while (abegin != aend && bbegin != bend) {
        if (*abegin == *bbegin) {
            intersection++;
            abegin++;
            bbegin++;
        } else if (*abegin < *bbegin) {
            abegin++;
        } else {
            bbegin++;
        }
    }
    return ((float) intersection) / ((float) n_a + n_b - intersection);
}

typedef std::pair<uint32_t, float>  ResultPair;

bool cmp_pairs(const ResultPair &a, const ResultPair &b) {
    return a.second > b.second
        || (a.second == b.second && a.first > b.first);
}

int main(int argc, char ** argv) {
    std::string path;
    size_t k = 1000;
    size_t sample_size = 0;

    if (argc == 2) {
        path = argv[1];
    } else if (argc == 4) {
        assert(argv[1] == "--sample");
        sample_size = atoi(argv[2]);
        path = argv[3];
    } else {
        std::cerr << "USAGE: StoreJaccard [--sample SIZE] <dataset>" << std::endl;
        return 1;
    }

    // load data
    std::cerr << "loading data from " << path << std::endl;
    H5Easy::File file(path, H5Easy::File::ReadWrite);
    std::vector<uint32_t> data = H5Easy::load<std::vector<uint32_t>>(file, "/train");
    std::vector<size_t> sizes = H5Easy::load<std::vector<size_t>>(file, "/size_train");
    size_t n = sizes.size();
    std::vector<size_t> offsets(n);
    size_t offset = 0;
    for (size_t i=0; i<n; i++) {
        offsets[i] = offset;
        offset += sizes[i];
    }

    size_t step = 1;
    if (sample_size > 0) {
        step = n / sample_size;
    }
    std::vector<size_t> indices;
    for (size_t i=0; i<n; i += step) {
        indices.push_back(i);
    }
    if (sample_size > 0) {
        H5Easy::dump(file, "/sample_indices", indices);
    }

    std::vector<std::vector<float>> top_similarities(n);
    std::vector<std::vector<uint32_t>> top_neighbors(n);
    std::vector<float> avg_similarities(n);

    size_t progress = 0;

    std::cerr << "computing similarities" << std::endl;
    // compute similarities
    #pragma omp parallel for schedule(dynamic)
    for (size_t h=0; h<indices.size(); h++) {
        size_t i = indices[h];
        std::vector<ResultPair> topk;
        for (size_t h=0; h<k; h++) {
            topk.emplace_back(std::numeric_limits<uint32_t>::max(), -1.0);
        }
        float sum_sim = 0.0;
        for (size_t j=0; j<n; j++) {
            if (i != j) {
                float similarity = jaccard(
                    data.begin() + offsets[i],
                    data.begin() + offsets[i] + sizes[i],
                    data.begin() + offsets[j],
                    data.begin() + offsets[j] + sizes[j]
                );
                sum_sim += similarity;
                if (similarity >= topk.front().second) {
                    topk.emplace_back(j, similarity);
                    std::push_heap(topk.begin(), topk.end(), cmp_pairs);
                    std::pop_heap(topk.begin(), topk.end(), cmp_pairs);
                    topk.pop_back();
                }
            }
        }
        avg_similarities[i] = sum_sim / n;

        std::sort_heap(topk.begin(), topk.end(), cmp_pairs);
        for (auto pair : topk) {
            // std::cerr << " [" << i << "] " << pair.first << " " << pair.second << std::endl;
            top_similarities[i].push_back(pair.second);
            top_neighbors[i].push_back(pair.first);
        }
        #pragma omp critical
        {
            if (++progress % 1000 == 0) {
                std::cerr << "completed " << progress << "/" << indices.size() << std::endl;
            }
        }
    }

    H5Easy::dump(file, "/top-1000-dists", top_similarities);
    H5Easy::dump(file, "/top-1000-neighbors", top_neighbors);
    H5Easy::dump(file, "/average_distance", avg_similarities);
}
