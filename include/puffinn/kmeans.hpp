#pragma once
#include "puffinn/dataset.hpp"

#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <cfloat>

namespace puffinn
{
    // TODO:
    // Can most likely avoid converting to floats and just use fixed point format
    // Better utilization of processer, perhaps parallelization
    // implement kmeans++ center initilization algorithm 
    // currently some centers have no points which are closest to them which is impossible when they are samples
    // and center at index 0 always have the majority which it shouldn't as they should be random


    /// Class for performing k-means clustering on a given dataset
    template <typename TFormat>
    class Kmeans
    {
        // reference to data contained in Index instance
        Dataset<TFormat> &dataset;
        DatasetDescription<TFormat> data_desc;
        const uint8_t K;

        // Pointers to start of array
        typename TFormat::Type* centers;
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
            centers = new typename TFormat::Type[K*data_desc.args] {};
            labels = new uint8_t[dataset.get_size()];
        }

        void fit()
        {
            std::cout << "fit called" << std::endl;
            init_centers();
            single_lloyd();
        }

    private:
        // samples K random points and uses those as starting centers
        void init_centers()
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
                    std::copy(sample, sample + data_desc.storage_len, centers + (c_i*data_desc.storage_len));
                    c_i++;
                }
            }
        }

        // Performs a single kmeans clustering 
        // centers are set to the member centers
        // Using the lloyd algorithm for clustering
        void single_lloyd() {

            float last_inertia;
            uint16_t iteration = 0;
            uint8_t labels[dataset.get_size()];
            std::fill(labels, labels+dataset.get_size(), K+1);

            while ((last_inertia-inertia) < tol && iteration < max_iter ) {
                last_inertia = inertia;
                inertia = setLabels(labels);
                show(labels, dataset.get_size());
                setNewCenters(labels);
                iteration++;
            }


        }

        // Sets the labels for all vectors returns the inertia for the current set of centers 
        float setLabels(uint8_t* const labels) {
            float sq_dist,
                  inertia,
                  distances[dataset.get_size()];
            std::fill(distances, distances+dataset.get_size(), 5.0);


            for (uint32_t i = 0; i < dataset.get_size(); i++) {
                //std::printf("0x%08x = %d = %f\n", i,i,distances[i]);
                for (uint8_t j = 0; j < K; j++) {
                    sq_dist = TFormat::distance(dataset[i], &centers[j], data_desc.args);
                    if (sq_dist < distances[i]){
                        distances[i]= sq_dist;
                        //std::printf("0x%08x\n", j);
                        labels[i] = j;
                    }
                }
                inertia += distances[i];
            }
            return inertia;
        }
        // Sets new centers according to average of
        // vectors belonging to the cluster
        void setNewCenters(uint8_t* const labels) {
            float new_centers[K*data_desc.args] = {};
            unsigned int counts[K] = {};
            for (unsigned int i = 0; i < dataset.get_size() ; i++) {
                TFormat::add_to_float(dataset[i], &new_centers[labels[i]*data_desc.args], data_desc.args);
                counts[labels[i]]++;
            }
            for (uint8_t cen = 0; cen < K; cen++) {
                for (unsigned int j = 0; j < data_desc.args; j++) {
                    float center_val = new_centers[cen*data_desc.args +j]/counts[cen];
                    std::printf("%f = %d = %f\n", new_centers[cen*data_desc.args+j],counts[cen],center_val);

                    std::cout << center_val << std::endl;
                    centers[cen*data_desc.args + j] = TFormat::to_16bit_fixed_point(center_val);
                }
            }
        }

        void show(uint8_t * arr, size_t size) {
            for (size_t i = 0; i < size; i++) {
                std::cout << (unsigned int)arr[i] << " ";
            }
            std::cout << std::endl;
        }
        void show() {
            for (int j = 0; j < 1; j++) {
                for (int i=0; i < data_desc.storage_len; i++) {
                    typename TFormat::Type fix_point = centers[data_desc.storage_len*j+i];
                    std::cout << TFormat::from_16bit_fixed_point(fix_point) << " ";
                }
                std::cout << std::endl;
            }
        }
    };
} // namespace puffin