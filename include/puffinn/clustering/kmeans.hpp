#pragma once

#include <vector>
#include <iostream>
#include <random>
#include <unordered_set>


namespace puffinn {


    /// Class for performing k-means clustering on a given dataset
    // T is type of a single data entry, K is number of clusters
    template <typename T, int K>
    class Kmeans 
    {
    private:
        std::vector<std::vector<T>> data;
    public:

        // Clusters are always floats
        std::vector<float> clusters[K];

        Kmeans(std::vector<std::vector<T>>& dataset) {
            this->data = dataset;
            // Check Dataset size vs number of clusters
        }

        Kmeans() {
            std::cout << "Initialized" << std::endl;

            // Check Dataset size vs number of clusters
        }

        void insert(const std::vector<T> vec) {
            data.push_back(vec);
        }

        void fit() {
            init_clusters();

        }

    private:
        // samples K random points and uses those as starting clusters
        void init_clusters() {
            std::random_device device;
            std::mt19937 rng(device());
            // uniform int distribution in range [0, dist.size()-1] to pick indicies for samples
            std::uniform_int_distribution<std::mt19937::result_type> distribution(0,data.size()-1);

            int8_t c_i = 0;
            std::unordered_set<int> used;
            while(c_i < K) {
                int sample_idx = distribution(rng);
                if (used.find(sample_idx) == used.end()) {
                    used.insert(sample_idx);
                    std::vector<float> sample(data[sample_idx].begin(), data[sample_idx].end());
                    clusters[c_i] = sample;
                    c_i++;
                }
            }

        }
    };
} // namespace puffin