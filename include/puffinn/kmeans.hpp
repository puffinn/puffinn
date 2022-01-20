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
        Dataset<TFormat> centroids;
        uint8_t* labels;

        Dataset<TFormat> sums;
        // contains sum of all vectors in clusters
        // contains count of vectors in each cluster
        unsigned int* counts;

        const float tol = 0.0001;
        const uint16_t max_iter = 300;
        float inertia = FLT_MAX;

    public:
        KMeans(Dataset<TFormat> &dataset, uint8_t K_clusters)
            : dataset(dataset),
              K(K_clusters),
              N(dataset.get_size()),
              vector_len(dataset.get_description().storage_len),
              centroids(vector_len, K),
              sums(vector_len,K)
        {
            std::cerr << "Kmeans info: \tN=" << N << "\tK=" << (unsigned int)K << std::endl;
            labels = new uint8_t[N];
            std::fill_n(labels, N, K+1);

            counts = new unsigned int[K] {};
        }

        ~KMeans() 
        {
            delete[] labels;
            delete[] counts;

        }

        void fit()
        {
            float distances[N];
            std::fill_n(distances, N, FLT_MAX);
            std::cerr << "fit called" << std::endl;
            // init_centers_random(); doesn't work
            init_centroids_kpp(distances);
            single_lloyd(distances);
        }

        typename TFormat::Type* getCentroid(size_t c_i) {
            return centroids[c_i];
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
                    std::copy(sample, sample + vector_len, centroids[c_i]);
                    c_i++;
                }
            }
        }

        void init_centroids_kpp(float * distances)
        {
            firstCentroid(distances);
            // 1 centroid is chosen
            for (size_t c_i = 1; c_i < K; c_i++) {
                // pick vector based on dists
                int vec = weightedRandomSTD(distances);
                // copy vector to centriod
                std::copy(dataset[vec], dataset[vec] + vector_len, centroids[c_i]);
                // compute all dists again
                calcDists(distances, c_i);
            }

        }

        void firstCentroid(float * distances)
        {

            // Pick random centroid uniformly
            auto &rand_gen = get_default_random_generator();
            std::uniform_int_distribution<unsigned int> random_idx(0, N-1);
            unsigned int sample_idx = random_idx(rand_gen);
            std::copy(dataset[sample_idx], dataset[sample_idx] + vector_len, centroids[0]);

            // Calc all dists to this centroid
            for (size_t i = 0; i < N; i++) {
                float dist = TFormat::distance(dataset[i], centroids[0], vector_len);
                distances[i] = dist;
                TFormat::add_assign(sums[0], dataset[i], vector_len);
                counts[0]++;
                labels[i] = 0;
            }

        }
        
        int weightedRandomSTD(float * distances)
        {       
            auto &rand_gen = get_default_random_generator();
            std::discrete_distribution<int> rng(distances, distances+N);
            float rn = rng(rand_gen);
            return rn;
        }
        // Performs a single kmeans clustering 
        // centers are set to the member centers
        // Using the lloyd algorithm for clustering
        void single_lloyd(float * distances) {

            float last_inertia;
            uint16_t iteration = 0;

            do
            {
                std::cerr << "lloyd iteration: " << iteration << std::endl;
                last_inertia = inertia;
                setLabels(distances);
                inertia = calcInertia(distances);
                show(labels, N);
                setNewCenters();
                iteration++;
                std::cerr << std::endl << std::endl;

            } while ((last_inertia-inertia) > tol && iteration < max_iter );
            
            std::cerr << "inertia diff: " << last_inertia << " - " <<  inertia << " = " << last_inertia - inertia << std::endl;


        }

        float calcInertia(float * distances)
        {
            float inertia = 0;
            for (size_t i = 0; i < N; i++) {
                inertia += distances[i];
            }
            return inertia;
        }
        // Calculates distances for all vectors to given centroid
        // Sets results in dists argument
        void calcDists(float * const dists, size_t c_i) 
        {
            // for every data entry
            for (size_t i = 0; i < N; i++) {
                float dist = TFormat::distance(dataset[i], centroids[c_i], vector_len);
                if (dist < dists[i]) {
                    updateState(i,c_i);
                    dists[i] = dist;
                }

            }

        }

        // Update Label for vector
        // update centroids sums for both old centroids and new assigned centroid
        // update counts for both centroids as well
        // i: index for vector
        // c_i: index for centroid
        void updateState(size_t i, size_t c_i)
        {
            TFormat::subtract_assign(sums[labels[i]], dataset[i], vector_len);
            TFormat::add_assign(sums[c_i], dataset[i], vector_len);
            counts[labels[i]]--;
            counts[c_i]++;
            labels[i] = c_i;
            return;

        }

        // Sets the labels for all vectors 
        void setLabels(float * distances) {
            for (size_t c_i = 0; c_i < K; c_i++) {
                calcDists(distances, c_i);
            }
            // debug
            std::cerr << "Distances for entries" << std::endl;
            show(distances, N);
            std::cerr << "Which leads to an inertia of " << inertia << std::endl;
        }

        // Sets new centers according to average of
        // vectors belonging to the cluster
        void setNewCenters() {
            std::cerr << "setNewCentroids start" << std::endl;
            showCentroids();
            std::copy(sums[0], sums[K-1] + vector_len, centroids[0]);
            // Average all centroids by the number of elements in cluster
            for (size_t c_i = 0; c_i < K; c_i++) {
                TFormat::divide_assign(centroids[c_i], counts[c_i], vector_len);
            }
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
                show(centroids[c_i], vector_len);
            }
        }
    };
} // namespace puffin