#pragma once
#include "puffinn/dataset.hpp"

#include <vector>
#include <iostream>
#include <random>
#include <unordered_set>

namespace puffinn
{

    /// Class for performing k-means clustering on a given dataset
    template <typename TFormat>
    class Kmeans
    {
        // reference to data contained in Index instance
        Dataset<TFormat> &dataset;
        DatasetDescription<TFormat> data_desc;
        const uint8_t K;

        // Pointers to start of array
        typename TFormat::Type* clusters;
        uint8_t* labels;

        const float tol = 0.0001;
        const uint16_t max_iter = 300;
        float inertia= FLT_MAX;

    public:
        Kmeans(Dataset<TFormat> &dataset, uint8_t K_clusters)
            : K(K_clusters),
              dataset(dataset),
              data_desc(dataset.get_description())
        {
            clusters = new typename TFormat::Type[K*data_desc.storage_len] {};
            labels = new uint8_t[dataset.get_size()];
        }

        void fit()
        {
            std::cout << "fit called" << std::endl;
            init_clusters();
            single_lloyd()
            float change = 10.0;
            while(change > threshhold)
        }

    private:
        // samples K random points and uses those as starting clusters
        void init_clusters()
        {
            // Try using kmeans++ initialization algorithm
            auto &rand_gen = get_default_random_generator();
            std::uniform_int_distribution<unsigned int> random_idx(0, dataset.get_size() - 1);

            int8_t c_i = 0;
            std::unordered_set<unsigned int> used;
            while(c_i < K) {
                unsigned int sample_idx = random_idx(rand_gen);
                if (used.find(sample_idx) == used.end()) {
                    used.insert(sample_idx);
                    typename TFormat::Type* sample = dataset[sample_idx];
                    std::copy(sample, sample + data_desc.storage_len, clusters + (c_i*data_desc.storage_len));
                    c_i++;
                }
            }
        }

        // Performs a single kmeans clustering 
        // clusters are set to the member clusters
        // Using the lloyd algorithm for clustering
        void single_llyod() {

            float last_inertia;
            uint16_t iteration;
            uint8_t labels[dataset.get_size()];

            while ((last_inertia-inertia) < tol && iteration < max_iter ) {
                last_inertia = inertia;
                inertia = calcLabels(labels);
                setNewClusters(labels);

            }


        }

        // Sets the labels for all vectors
        // returns the inertia for the current set of clusters 
        float calcLabels(uint8_t* const labels) {
            float sq_dist,
                  inertia,
                  distances[dataset.get_size()] = {5.0};


            for (uint32_t i = 0; i < dataset.get_size(); i++) {
                for (uint8_t j = 0; j < K; j++) {
                    sq_dist = TFormat::distance(dataset[i], clusters[j], data_desc.args);
                    if (sq_dist < distances[i]){
                        distances[i]= sq_dist;
                        labels[i] = j;
                    }
                }
                inertia += distances[i];
            }
            return inertia;
        }
        void setNewClusters(uint8_t* const labels) {
            // TODO: whole method
        }

        void show() {
            for (int j = 0; j < 1; j++) {
                for (int i=0; i < data_desc.storage_len; i++) {
                    typename TFormat::Type fix_point = clusters[data_desc.storage_len*j+i];
                    std::cout << TFormat::from_16bit_fixed_point(fix_point) << " ";
                }
                std::cout << std::endl;
            }
        }
    };
} // namespace puffin