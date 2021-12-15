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
        uint8_t K;
        // Pointer to start of array
        typename TFormat::Type* clusters;

    public:
        Kmeans(Dataset<TFormat> &dataset, uint8_t K_clusters)
            : K(K_clusters),
              dataset(dataset),
              data_desc(dataset.get_description())
        {
            clusters = new typename TFormat::Type[K*data_desc.storage_len] {};
        }

        void fit()
        {
            std::cout << "fit called" << std::endl;
            init_clusters();
        }

    private:
        // samples K random points and uses those as starting clusters
        void init_clusters()
        {
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