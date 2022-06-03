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
    if (argc != 2) {
        std::cerr << "USAGE: StoreJaccard <dataset>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    size_t k = 1000;

    // load data
    H5Easy::File file(path, H5Easy::File::ReadOnly);
    std::vector<uint32_t> data = H5Easy::load<std::vector<uint32_t>>(file, "/train");
    std::vector<size_t> sizes = H5Easy::load<std::vector<size_t>>(file, "/size_train");
    size_t n = sizes.size();
    std::vector<size_t> offsets(n);
    size_t offset = 0;
    for (size_t i=0; i<n; i++) {
        offsets[i] = offset;
        offset += sizes[i];
    }

    std::vector<std::vector<float>> top_similarities(n);
    std::vector<std::vector<uint32_t>> top_neighbors(n);
    std::vector<float> avg_similarities(n);

    // size_t i=n-2, j=n-3;
    // for (auto it=data.begin() +offsets[i]; it != data.begin() + offsets[i] + sizes[i]; it++) {
    //     std::cerr << " "  << *it;
    // }
    // std::cerr << std::endl;
    // for (auto it=data.begin() +offsets[j]; it != data.begin() + offsets[j] + sizes[j]; it++) {
    //     std::cerr << " "  << *it;
    // }
    // std::cerr << std::endl;
    // float test = jaccard(
    //     data.begin() + offsets[i],
    //     data.begin() + offsets[i] + sizes[i],
    //     data.begin() + offsets[j],
    //     data.begin() + offsets[j] + sizes[j]
    // );
    // std::cerr << test << std::endl;
    // return 0;

    // compute similarities
    #pragma omp parallel for
    for (size_t i=0; i<n; i++) {
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
    }

    H5Easy::dump(file, "/top-1000-dists", top_similarities);
    H5Easy::dump(file, "/top-1000-neighbors", top_neighbors);
    H5Easy::dump(file, "/average_distance", avg_similarities);
}
