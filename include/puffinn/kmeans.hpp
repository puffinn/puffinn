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
    // Handling of other data-formats
    // Investigate parallelization within the clustering for performance gain
    // Implement kmeans++ center initilization algorithm 
    // Manage padding when using avx2 and vectors are divided into M sections

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
            std::cerr << "Kmeans info: \tN=" << N << "\tK=" << (unsigned int)K << std::endl;
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
            std::cerr << "fit called" << std::endl;
            init_centers_random();
            single_lloyd();
        }

        typename TFormat::Type* getCentroid(size_t index) {
            return centroids + (index*vector_len);
        }

    private:
        // samples K random points and uses those as starting centers
        void init_centers_random()
        {
            // Try using kmeans++ initialization algorithm
            std::cerr << "Init random centers" << std::endl;
            auto &rand_gen = get_default_random_generator();
            std::uniform_int_distribution<unsigned int> random_idx(0, N-1);

            int8_t c_i = 0;
            std::unordered_set<unsigned int> used;
            while(c_i < K) {
                unsigned int sample_idx = random_idx(rand_gen);
                if (used.find(sample_idx) == used.end()) {
                    used.insert(sample_idx);
                    typename TFormat::Type* sample = dataset[sample_idx];
                    std::copy(sample, sample + vector_len, centroids + (c_i*vector_len));
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

            do
            {
                std::cerr << "lloyd iteration: " << iteration << std::endl;
                last_inertia = inertia;
                inertia = setLabels();
                show(labels, N);
                setNewCenters();
                iteration++;
                std::cerr << std::endl << std::endl;

            } while ((last_inertia-inertia) > tol && iteration < max_iter );
            
            std::cerr << "inertia diff: " << last_inertia << " - " <<  inertia << " = " << last_inertia - inertia << std::endl;


        }

        // Sets the labels for all vectors returns the inertia for the current set of centers 
        float setLabels() {
            float inertia = 0,
                  distances[N];
            std::fill_n(distances, N, FLT_MAX);

            // for every data entry
            for (size_t i = 0; i < N; i++) {
                // for every centroid
                for (size_t c_i = 0; c_i < K; c_i++) {
                    float dist = TFormat::distance(dataset[i], centroids + (c_i*vector_len), vector_len);
                    if (dist < distances[i]){
                        distances[i] = dist;
                        labels[i] = c_i;
                    }
                }
                inertia += distances[i];
            }
            std::cerr << "Distances for entries" << std::endl;
            show(distances, N);
            std::cerr << "Which leads to an inertia of " << inertia << std::endl;
            return inertia;
        }

        // Sets new centers according to average of
        // vectors belonging to the cluster
        void setNewCenters() {
            std::cerr << "setNewCentroids start" << std::endl;
            showCentroids();
            typename TFormat::Type new_centroids[K*vector_len] = {};
            unsigned int counts[K] = {};
            // Add all vectors in a cluster
            for (size_t i = 0; i < N; i++) {
                typename TFormat::Type* new_centroid_start = new_centroids + (labels[i]*vector_len);
                TFormat::add_assign(new_centroid_start, dataset[i], vector_len);
                counts[labels[i]]++;
            }
            // Average all centroids by the number of elements in cluster
            for (size_t c_i = 0; c_i < K; c_i++) {
                typename TFormat::Type* new_centroid_start = new_centroids + (c_i*vector_len);
                TFormat::divide_assign(new_centroid_start, counts[c_i], vector_len);
            }
            // copy to class variable centroids
            std::copy(new_centroids, new_centroids + (K*vector_len), centroids); 
            std::cerr << "setNewCentroids end" << std::endl;
            showCentroids();
        }

        void show(uint8_t * arr, size_t size) {
            for (size_t i = 0; i < size; i++) {
                std::cerr << (unsigned int)arr[i] << " ";
            }
            std::cerr << std::endl;
        }

        void show(typename TFormat::Type* arr, size_t size) {
            for (size_t i = 0; i < size; i++) {
                std::cerr << arr[i] << " ";
            }
            std::cerr << std::endl;
        }
        void showCentroids() {
            for (size_t c_i = 0; c_i < K; c_i++) {
                std::cerr << "Centroid " << c_i << " ";
                show(centroids + (c_i*vector_len), vector_len);
            }
        }
    };
} // namespace puffin