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
    class KMeans
    {
        // reference to data contained in Index instance
        Dataset<TFormat> &dataset;
        const uint8_t K;
        const size_t N;
        const size_t vector_len;

        // Pointers to start of array
        typename TFormat::Type* centroids;
        uint8_t* labels;

        const float tol = 0.0001;
        const uint16_t max_iter = 300;
        float inertia = FLT_MAX;

    public:
        KMeans(Dataset<TFormat> &dataset, uint8_t K_clusters)
            : dataset(dataset),
              K(K_clusters),
              N(dataset.get_size()),
              vector_len(dataset.get_description().storage_len)
        {
            std::cout << "Kmeans info: \tN=" << N << "\tK=" << (unsigned int)K << std::endl;
            centroids = new typename TFormat::Type[K*vector_len] {};
            labels = new uint8_t[N];
            std::fill_n(labels, N, K+1);
        }

        ~KMeans() 
        {
            delete[] centroids;
            delete[] labels;

        }

        void fit()
        {
            std::cout << "fit called" << std::endl;
            init_centers_random();
            single_lloyd();
        }

    private:
        // samples K random points and uses those as starting centers
        void init_centers_random()
        {
            // Try using kmeans++ initialization algorithm
            std::cout << "Init random centers" << std::endl;
            auto &rand_gen = get_default_random_generator();
            std::uniform_int_distribution<unsigned int> random_idx(0, N-1);

            int8_t c_i = 0;
            std::unordered_set<unsigned int> used;
            while(c_i < K) {
                unsigned int sample_idx = random_idx(rand_gen);
                if (used.find(sample_idx) == used.end()) {
                    used.insert(sample_idx);
                    typename TFormat::Type* sample = dataset[sample_idx];
                    std::copy(sample, &sample[vector_len], &centroids[c_i++]);
                }
            }
        }

        // Performs a single kmeans clustering 
        // centers are set to the member centers
        // Using the lloyd algorithm for clustering
        void single_lloyd() {

            float last_inertia;
            uint16_t iteration = 0;

            do
            {
                std::cout << "lloyd iteration: " << iteration;
                last_inertia = inertia;
                inertia = setLabels(labels);
                std::cout << " with inertia: " << inertia << std::endl;
                show(labels, N);
                setNewCenters(labels);
                iteration++;

            } while ((last_inertia-inertia) > tol && iteration < max_iter );
            
            std::cout << "inertia diff: " << last_inertia << " - " <<  inertia << " = " << last_inertia - inertia << std::endl;


        }

        // Sets the labels for all vectors returns the inertia for the current set of centers 
        float setLabels(uint8_t* const labels) {
            float inertia = 0,
                  distances[dataset.get_size()];
            std::fill_n(distances, N, FLT_MAX);


            // for every data entry
            for (size_t i = 0; i < N; i++) {
                //std::printf("0x%08x = %d = %f\n", i,i,distances[i]);
                // for every centroid
                for (size_t c_i = 0; c_i < K; c_i++) {
                    float dist = TFormat::distance(dataset[i], &centroids[c_i], vector_len);
                    //std::cout << "index: " << i << " c_i: " << c_i << " dist=" << dist << std::endl;
                    if (dist < distances[i]){

                        distances[i] = dist;
                        //std::printf("0x%08x\n", j);
                        labels[i] = c_i;
                    }
                }
                inertia += distances[i];
            }
            return inertia;
        }
        // Sets new centers according to average of
        // vectors belonging to the cluster
        void setNewCenters(uint8_t* const labels) {
            // Doesn't seem to work
            std::cout << "setNewCenters start" << std::endl;
            typename TFormat::Type new_centroids[K*vector_len];
            unsigned int counts[K] = {};
            // Add all vectors in a cluster
            for (size_t i = 0; i < N; i++) {
                typename TFormat::Type* new_centroid_start = &new_centroids[labels[i]*vector_len];
                TFormat::add_assign(new_centroid_start, dataset[i], vector_len);
                counts[labels[i]]++;
            }
            // Average all centroids by the number of elements in cluster
            for (size_t c_i = 0; c_i < K; c_i++) {
                typename TFormat::Type* new_centroid_start = &new_centroids[c_i*vector_len];
                TFormat::divide_assign(new_centroid_start, counts[c_i], vector_len);
                std::cout << 
                show(new_centroid_start, 2);
            }
            // copy to class variable centroids
            std::copy(new_centroids, &new_centroids[K*vector_len], centroids); 
        }

        void show(uint8_t * arr, size_t size) {
            for (size_t i = 0; i < size; i++) {
                std::cout << (unsigned int)arr[i] << " ";
            }
            std::cout << std::endl;
        }

        void show(typename TFormat::Type* arr, size_t size) {
            for (size_t i = 0; i < size; i++) {
                std::cout << arr[i] << " ";
            }
            std::cout << std::endl;
        }
    };
} // namespace puffin