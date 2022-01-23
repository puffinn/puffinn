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
        struct RunData {
            Dataset<TFormat> centroids, sums;
            unsigned int* counts;
            uint8_t* labels;
            float* distances;
            const size_t N;

            RunData(size_t N, size_t K, size_t vector_len)
            : centroids(vector_len, K),
              sums(vector_len, K),
              N(N)
            {
                counts = new unsigned int[K] {};
                labels = new uint8_t[N] {};
                distances = new float[N];
                resetDists();
            }

            ~RunData() {
                delete[] counts;
                delete[] labels;
                delete[] distances;
            }
            inline void resetDists() {
                std::fill_n(distances, N, FLT_MAX);
            }
        };

        static constexpr float TOL = 0.0001f;
        static const uint16_t MAX_ITER = 300;
        static const uint8_t N_RUNS = 10;

        const uint8_t K;
        const size_t N;
        const size_t vector_len;
        // reference to data contained in Index instance
        // centroids is for the current run and gb_centroids is for the global best
        Dataset<TFormat> &dataset,
                          gb_centroids;

        uint8_t* gb_labels;

        // We don't need centroid, labels, sums, counts after fit has been called
        // They takeup quite a lot of space

        // contains sum of all vectors in clusters
        // contains count of vectors in each cluster

        float gb_inertia = FLT_MAX;

    public:
        KMeans(Dataset<TFormat> &dataset, uint8_t K_clusters)
            : K(K_clusters),
              N(dataset.get_size()),
              vector_len(dataset.get_description().storage_len),
              dataset(dataset),
              gb_centroids(vector_len, K)
        {
            std::cerr << "Kmeans info: \tN=" << N << "\tK=" << (unsigned int)K << std::endl;
            gb_labels = new uint8_t[N];
        }

        ~KMeans() 
        {
            delete[] gb_labels;
        }

        void fit()
        {
            std::cerr << "fit called" << std::endl;
            // init_centers_random(); doesn't work
            for(uint8_t run=0; run < N_RUNS; run++) {
                struct RunData rd(N, K, vector_len);
                float run_inertia = single_kmeans(rd);
                if (run_inertia < gb_inertia) {
                    // New run is the currently best, thus overwrite gb variables
                    gb_inertia = run_inertia;
                    // gb_centroids = rd.centroids;
                    std::copy(rd.labels, rd.labels+N, gb_labels);
                }
            }
        }

        typename TFormat::Type* getCentroid(size_t c_i) {
            return gb_centroids[c_i];
        }

    private:

        float single_kmeans(struct RunData& rd)
        {
            std::cerr << "single_kmeans called" << std::endl;
            // init_centers_random(); doesn't work
            init_centroids_kpp(rd);
            return lloyd(rd);

        }
        // samples K random points and uses those as starting centers
        void init_centers_random(Dataset<TFormat> &cen)
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
                    std::copy(sample, sample + vector_len, cen[c_i]);
                    c_i++;
                }
            }
        }

        void init_centroids_kpp(struct RunData& rd)
        {
            showCentroids(rd.centroids);
            firstCentroid(rd);
            // 1 centroid is chosen
            for (size_t c_i = 1; c_i < K; c_i++) {
                // pick vector based on distances
                int vec = weightedRandomSTD(rd.distances);
                // copy vector to centriod
                std::copy(dataset[vec], dataset[vec] + vector_len, rd.centroids[c_i]);
                // compute all distances again
                calcDists(rd, c_i);
            }
            showCentroids(rd.centroids);

        }

        void firstCentroid(struct RunData& rd)
        {
            // Pick random centroid uniformly
            auto &rand_gen = get_default_random_generator();
            std::uniform_int_distribution<unsigned int> random_idx(0, N-1);
            unsigned int sample_idx = random_idx(rand_gen);
            std::copy(dataset[sample_idx], dataset[sample_idx] + vector_len, rd.centroids[0]);

            // Calc all distances to this centroid
            for (size_t i = 0; i < N; i++) {
                float dist = TFormat::distance(dataset[i], rd.centroids[0], vector_len);
                rd.distances[i] = dist;
                TFormat::add_assign(rd.sums[0], dataset[i], vector_len);
                rd.counts[0]++;
                rd.labels[i] = 0;
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
        // centroids are set to the member centroids
        // Using the lloyd algorithm for clustering
        float lloyd(struct RunData& rd) {

            float inertia = FLT_MAX, last_inertia;
            uint16_t iteration = 0;

            do
            {
                rd.resetDists();
                std::cerr << "lloyd iteration: " << iteration << std::endl;
                last_inertia = inertia;
                setLabels(rd);
                inertia = calcInertia(rd.distances);
                std::cerr << "Which leads to an inertia of " << inertia << std::endl;
                show(rd.labels, N);
                setNewCenters(rd);
                iteration++;
                std::cerr << std::endl << std::endl;

            } while ((last_inertia-inertia) > TOL && iteration < MAX_ITER );
            
            std::cerr << "inertia diff: " << last_inertia << " - " <<  inertia << " = " << last_inertia - inertia << std::endl;
            return inertia;


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
        // Sets results in distances argument
        void calcDists(struct RunData& rd, size_t c_i) 
        {
            // for every data entry
            for (size_t i = 0; i < N; i++) {
                float dist = TFormat::distance(dataset[i], rd.centroids[c_i], vector_len);
                if (dist < rd.distances[i]) {
                    updateState(rd, i, c_i);
                    rd.distances[i] = dist;
                }

            }

        }

        // Update Label for vector
        // update centroids sums for both old centroids and new assigned centroid
        // update counts for both centroids as well
        // i: index for vector
        // c_i: index for centroid
        void updateState(struct RunData& rd, size_t i, size_t c_i)
        {
            if (rd.labels[i] == c_i) return;
            TFormat::subtract_assign(rd.sums[rd.labels[i]], dataset[i], vector_len);
            TFormat::add_assign(rd.sums[c_i], dataset[i], vector_len);
            rd.counts[rd.labels[i]]--;
            rd.counts[c_i]++;
            rd.labels[i] = c_i;
            return;

        }

        // Sets the labels for all vectors 
        void setLabels(struct RunData& rd) {
            for (size_t c_i = 0; c_i < K; c_i++) {
                calcDists(rd, c_i);
            }
            // debug
            std::cerr << "Distances for entries" << std::endl;
            show(rd.distances, N);
        }

        // Sets new centers according to average of
        // vectors belonging to the cluster
        void setNewCenters(struct RunData& rd) {
            std::cerr << "setNewCentroids start" << std::endl;
            showCentroids(rd.centroids);
            std::copy(rd.sums[0], rd.sums[K-1] + vector_len, rd.centroids[0]);
            // Average all centroids by the number of elements in cluster
            for (size_t c_i = 0; c_i < K; c_i++) {
                TFormat::divide_assign(rd.centroids[c_i], rd.counts[c_i], vector_len);
            }
            std::cerr << "setNewCentroids end" << std::endl;
            showCentroids(rd.centroids);
        }

        void show(uint8_t * arr, size_t size) 
        {
            for (size_t i = 0; i < size; i++) {
                std::cerr << (unsigned int)arr[i] << " ";
            }
            std::cerr << std::endl;
        }

        void show(typename TFormat::Type* arr, size_t size) 
        {
            for (size_t i = 0; i < size; i++) {
                std::cerr << arr[i] << " ";
            }
            std::cerr << std::endl;
        }

        void showCentroids()
        {
            showCentroids(gb_centroids);
        }

        void showCentroids(Dataset<TFormat> &cen) {
            for (size_t c_i = 0; c_i < K; c_i++) {
                std::cerr << "Centroid " << c_i << ": ";
                show(cen[c_i], vector_len);
            }
        }
    };
} // namespace puffin