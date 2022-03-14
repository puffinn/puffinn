#pragma once
#include <chrono>
#include <map>
#include <unordered_map>
#include <vector>

#include "puffinn/typedefs.hpp"

const bool PUFFINN_PERFORMANCE = true;
const bool PUFFINN_PERFORMANCE_TIME = true;

namespace puffinn {
    const size_t NUM_TIMED_COMPUTATIONS = 11;
    // Indented to match subgroups
    enum class Computation {
        Total,
            Hashing,
            Sketching,
            Search,
                SearchInit,
                    CreateQuery, // Also includes hashing
                ReducePrefix, // Find candidates
                Filtering,
                Consider,
                    MaxbufferFilter,
                CheckTermination
    };

    struct QueryMetrics {
        unsigned int distance_computations;
        unsigned int candidates;
        unsigned int considered_maps;
        unsigned int hash_length;
        double time[NUM_TIMED_COMPUTATIONS];

        double get_time(Computation computation) const {
            return time[static_cast<int>(computation)];
        }

    };

    // A globally accessible structure to store performance metrics in.
    class PerformanceMetrics {
        std::vector<QueryMetrics> queries;

        // Stores last time start_timer was called.
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time[NUM_TIMED_COMPUTATIONS];

        QueryMetrics& current_query() {
            return queries[queries.size()-1];
        }

    public:
        PerformanceMetrics() {
            clear();
        }

        void clear() {
            queries.clear();
            // Add an empty query so that methods can be called without starting a query.
            // This can happen in tests.
            queries.push_back(QueryMetrics());
            new_query();
        }

        void new_query() {
            if (PUFFINN_PERFORMANCE || PUFFINN_PERFORMANCE_TIME) {
                queries.push_back(QueryMetrics());
            }
        }

        void add_distance_computations(unsigned int count) {
            if (PUFFINN_PERFORMANCE) {
                current_query().distance_computations += count;
            }
        }

        void add_candidates(unsigned int count) {
            if (PUFFINN_PERFORMANCE) {
                current_query().candidates += count;
            }
        }

        void set_hash_length(unsigned int len) {
            if (PUFFINN_PERFORMANCE) {
                current_query().hash_length = len;
            }
        }

        void set_considered_maps(unsigned int count) {
            if (PUFFINN_PERFORMANCE) {
                current_query().considered_maps = count;
            }
        }

        std::vector<QueryMetrics> get_query_metrics() const {
            std::vector<QueryMetrics> res;
            for (size_t i=1; i < queries.size(); i++) {
                res.push_back(queries[i]);
            }
            return res;
        }

        double get_total_time(Computation computation) {
            double res = 0;
            for (auto& q : queries) {
                res += q.get_time(computation);
            }
            return res;
        }

        // Start a timer whose result is stored using store_time.
        void start_timer(Computation computation) {
            if (PUFFINN_PERFORMANCE_TIME) {
                start_time[static_cast<int>(computation)] = std::chrono::high_resolution_clock::now();
            }
        }

        // Store that the given computation has taken the time since last call to start_timer().
        void store_time(Computation computation) {
            if (PUFFINN_PERFORMANCE_TIME) {
                int computation_idx = static_cast<int>(computation);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end_time-start_time[computation_idx];
                current_query().time[computation_idx] += duration.count();
            }
        }
    };

    PerformanceMetrics g_performance_metrics;
}

