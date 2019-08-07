#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/filterer.hpp"
#include "puffinn/hash_source/hash_source.hpp"
#include "puffinn/hash_source/independent.hpp"
#include "puffinn/maxbuffer.hpp"
#include "puffinn/prefixmap.hpp"
#include "puffinn/typedefs.hpp"

#include <omp.h>
#include <cassert>
#include <memory>
#include <vector>

namespace puffinn {
    /// Approaches to filtering candidates.
    enum class FilterType {
        /// The most optimized and recommended approach, which stops
        /// shortly after the required expected recall has been achieved.
        Default,
        /// A simple approach without sketching.
        /// Use this if it is very important that the *expected* recall is above the given treshold.
        /// However it currently looks at every table in the internal structure
        /// before checking whether the recall target has been achieved. 
        None,
        /// A simple approach which mirrors ``None``, but with filtering.
        /// It is only intended to be used to fairly assess the impact of sketching on the result. 
        Simple
    };

    /// An index constructed over a dataset which supports approximate
    /// near-neighbor queries for a specific similarity measure.
    /// 
    /// Basic usage consists of using the ``insert``, ``rebuild`` and ``search`` 
    /// methods in that order.
    /// These methods are the only ones in the library that need to be called during typical use.
    /// 
    /// The ``Index`` is generic over the similarity measure and two LSH families, one used for searching and
    /// one used for the following filtering step.
    /// The LSH families default to good choices for the similarity measure
    /// and should usually not be explicitly set. 
    ///
    /// @param TSim The similarity measure. Currently ``CosineSimilarity`` and ``JaccardSimilarity`` are supported.
    /// Depending on the similarity measure, points are stored internally using different ``Format``s.
    /// The ``Format`` specifies which types of input are supported.
    /// @param THash The family of Locality-Sensitive hash functions used to
    /// search for near neighbor candidates.
    /// Defaults to a family chosen by the similarity measure. 
    /// @param TSketch The family of 1-bit Locality-Sensitive hash functions
    /// used to further filter candidates.
    /// Defaults to a family chosen by the similarity measure.
    template <
        typename TSim,
        typename THash = typename TSim::DefaultHash,
        typename TSketch = typename TSim::DefaultSketch
    >
    class Index {
        Dataset<typename TSim::Format> dataset;
        // Hash tables used by LSH.
        std::vector<PrefixMap<THash>> lsh_maps;
        std::unique_ptr<HashSource<THash>> hash_source;
        // Container of sketches. Also needs to be reset.
        Filterer<TSketch> filterer;

        // Number of bytes allowed to be used.
        uint64_t memory_limit;
        // Number of values inserted the last time rebuild was called.
        uint32_t last_rebuild = 0;
        // Construction of the hash source is delayed until the
        // first rebuild so that we know how many tables are at most used.
        std::unique_ptr<HashSourceArgs<THash>> hash_args;

        constexpr static IndependentHashArgs<THash> DEFAULT_HASH_SOURCE
            = IndependentHashArgs<THash>();
        constexpr static IndependentHashArgs<TSketch> DEFAULT_SKETCH_SOURCE
            = IndependentHashArgs<TSketch>();

    public:
        /// Construct an empty index.
        ///
        /// @param dataset_args Arguments specifying how the dataset should be stored,
        /// depending on the format of the similarity measure.
        /// When using ``CosineSimilarity``, it specifies the dimension that all vectors must have.
        /// When using ``JaccardSimilarity``, it specifies the universe size. All tokens must be
        /// integers between 0, inclusive, and the paramter, exclusive.
        /// @param memory_limit The number of bytes of memory that the index is permitted to use.
        /// Using more memory almost always means that queries are more efficient.
        /// @param hash_args Arguments used to construct the source from which hashes are drawn.
        /// This also includes arguments that are specific to the hash family specified in ``THash``.
        /// It is recommended to use the default value.
        /// @param sketch_args Similar to ``hash_args``, but for the hash family specified in ``TSketch``.
        /// It is recommended to use the default value.
        Index(
            typename TSim::Format::Args dataset_args,
            uint64_t memory_limit,
            const HashSourceArgs<THash>& hash_args = DEFAULT_HASH_SOURCE,
            const HashSourceArgs<TSketch>& sketch_args = DEFAULT_SKETCH_SOURCE
        )
          : dataset(Dataset<typename TSim::Format>(dataset_args)),
            filterer(
                sketch_args.build(
                    dataset.get_description(),
                    NUM_SKETCHES,
                    NUM_FILTER_HASHBITS)),
            memory_limit(memory_limit),
            hash_args(hash_args.copy())
        {
            static_assert(
                std::is_same<TSim, typename THash::Sim>::value
                && std::is_same<TSim, typename TSketch::Sim>::value);
        }

        /// Insert a value into the index.
        ///
        /// Before the value can be found using the ``search`` method,
        /// ``rebuild`` must be called.
        /// 
        /// @param value The value to insert.
        /// The type must be supported by the format used by ``TSim``.
        template <typename T>
        void insert(const T& value) {
            dataset.insert(value);
            // Dont insert into the hash tables as it would be in linear time.
        }

        /// Rebuild the index using the currently inserted points.
        /// 
        /// This can be done in parallel.
        ///
        /// @param num_threads The number of threads to use.
        /// If set to zero, it will use as many threads as there are cores. 
        void rebuild(unsigned int num_threads = 0) {
            if (num_threads != 0) {
                omp_set_num_threads(num_threads);
            }

            // Upper bound on the size of the data stored inside hash buckets.
            unsigned int table_size =
                dataset.get_size() * (sizeof(uint32_t) + sizeof(LshDatatype))
                + sizeof(PrefixMap<THash>);
            unsigned int dataset_bytes =
                dataset.get_capacity() * dataset.get_description().storage_len
                * sizeof(typename TSim::Format::Type);
            auto necessary_bytes = dataset_bytes;
            // Not enough memory for at least one table
            if (memory_limit < necessary_bytes+table_size) {
                throw std::invalid_argument("insufficient memory");
            }
            unsigned int num_tables = (memory_limit-necessary_bytes)/table_size;

            // if rebuild has been called before
            if (hash_source) {
                // Resize the number of tables
                while (lsh_maps.size() > num_tables) {
                    // Discard the last tables. Since values are never deleted,
                    // the number of tables is not going to increase again.
                    lsh_maps.pop_back();
                }
            } else {
                hash_source = hash_args->build(
                    dataset.get_description(),
                    num_tables,
                    MAX_HASHBITS);
                // Construct the prefixmaps.
                for (unsigned int repetition=0; repetition < num_tables; repetition++) {
                    lsh_maps.emplace_back(this->hash_source->sample(), MAX_HASHBITS);
                }
            }

            // Compute sketches for the new vectors.
            filterer.add_sketches(dataset, last_rebuild);

            // Compute hashes for the new vectors in order, so that caching works.
            // Hash a vector in all the different ways needed.
            for (size_t idx=last_rebuild; idx < dataset.get_size(); idx++) {
                this->hash_source->reset(dataset[idx]);
                // Only parallelize if this step is computationally expensive.
                if (hash_source->precomputed_hashes()) {
                    for (auto& map : lsh_maps) {
                        map.insert(idx);
                    }
                } else {
                    #pragma omp parallel for
                    for (size_t map_idx = 0; map_idx < lsh_maps.size(); map_idx++) {
                        lsh_maps[map_idx].insert(idx);
                    }
                }
            }

            for (size_t map_idx = 0; map_idx < lsh_maps.size(); map_idx++) {
                lsh_maps[map_idx].rebuild();
            }
            last_rebuild = dataset.get_size();
        }

        /// Search for the approximate ``k`` nearest neighbors to a query.
        ///
        /// @param query The query value.
        /// It follows the same constraints as when inserting a value.
        /// @param k The number of neighbors to search for.
        /// @param recall The expected recall of the result.
        /// Each of the nearest neighbors has at least this probability
        /// of being found in the first phase of the algorithm.
        /// However if sketching is used, the probability of the neighbor being returned might be slightly lower. 
        /// This is given as a number between 0 and 1.
        /// @param filter_type The approach used to filter candidates.
        /// Unless the expected recall needs to be strictly above the ``recall`` parameter, the default should be used.
        /// @return The indices of the ``k`` nearest found neighbors.
        /// Indices are assigned incrementally to each point in the order they are inserted into the dataset, starting at 0.
        /// The result is ordered so that the most similar neighbor is first.
        template <typename T>
        std::vector<uint32_t> search(
            const T& query,
            unsigned int k,
            float recall,
            FilterType filter_type = FilterType::Default
        ) {
            if (dataset.get_size() < 100) {
                // Due to optimizations values near the edges in prefixmaps are discarded.
                // When there are fewer total values than SEGMENT_SIZE, all values will be skipped.
                // However at that point, brute force is likely to be faster regardless.
                return search_bf(query, k);
            }
            g_performance_metrics.new_query();
            g_performance_metrics.start_timer(Computation::Total);
            auto desc = dataset.get_description();

            // Ensure that the vec is kept alive for the entire scope
            auto stored_query = to_stored_type<typename TSim::Format>(query, desc);

            MaxBuffer maxbuffer(k);
            g_performance_metrics.start_timer(Computation::Hashing);
            auto hash_query = to_stored_type<typename THash::Sim::Format>(query, desc);
            hash_source->reset(hash_query.get());
            g_performance_metrics.store_time(Computation::Hashing);
            g_performance_metrics.start_timer(Computation::Sketching);
            auto sketch_query = to_stored_type<typename TSketch::Sim::Format>(query, desc);
            filterer.reset(sketch_query.get());
            g_performance_metrics.store_time(Computation::Sketching);

            g_performance_metrics.start_timer(Computation::Search);
            switch (filter_type) {
                case FilterType::None:
                    search_maps_no_filter(stored_query.get(), maxbuffer, recall);
                    break;
                case FilterType::Simple:
                    search_maps_simple_filter(stored_query.get(),
                    maxbuffer,
                    recall);
                    break;
                default:
                    search_maps(stored_query.get(), maxbuffer, recall);
            }
            g_performance_metrics.store_time(Computation::Search);

            auto res = maxbuffer.best_indices();
            g_performance_metrics.store_time(Computation::Total);
            return res;
        }

        /// Search for the k nearest neighbors to a query by 
        /// computing the similarity of each inserted value.
        ///
        /// ``rebuild`` does not need to be called before a point is considered. 
        /// 
        /// @param query The query value.
        /// It follows the same constraints as when inserting a value. 
        /// @param k The number of neighbors to search for.
        /// @return The indices of the ``k`` nearest neighbors.
        /// Indices are assigned incrementally to each point in the order they are inserted into the dataset, starting at 0.
        /// The result is ordered so that the most similar neighbor is first.
        template <typename T>
        std::vector<unsigned int> search_bf(
            const T& query,
            unsigned int k
        ) const {
            auto stored = to_stored_type<typename TSim::Format>(
                query, dataset.get_description());
            MaxBuffer res(k);
            for (size_t i=0; i < dataset.get_size(); i++) {
                float sim = TSim::compute_similarity(
                    stored.get(),
                    dataset[i],
                    dataset.get_description());
                res.insert(i, sim);
            }
            std::vector<uint32_t> res_indices;
            for (auto p : res.best_entries()) {
                res_indices.push_back(p.first);
            }
            return res_indices;
        }

        // Retrieve the number of inserted vectors.
        unsigned int get_size() const {
            return dataset.get_size();
        }

        // Retrieve the number of tables used internally.
        size_t get_repetitions() const {
            return lsh_maps.size();
        }

    private:
        // Size of buffer of 4element segments to consider at once.
        const static int RING_SIZE = NUM_SKETCHES;

        struct SearchBuffers {
            // Filler range, used as some valid range data for the one-beyond-end
            // index when searching
            uint32_t range_end_filler[2*RING_SIZE*4] = {0};

            // Data for each range. One longer than the number of tables to always allow
            // access to the next range.

            // Storage for each range. Each table gives one range of elements to consider.
            size_t num_ranges = 0;
            // Empty ranges are discarded.
            // +1 to always allow safe access to the next range
            std::unique_ptr<std::pair<const uint32_t*, const uint32_t*>[]> ranges;
            // For each range, which table it was taken from.
            std::unique_ptr<uint_fast32_t[]> table_indices;

            // Stores the range of values that have already been considered.
            // Before a table can be used, the initial point is found through binary search.
            std::vector<PrefixMapQuery> query_objects;

            SearchBuffers(const std::vector<PrefixMap<THash>>& maps) {
                g_performance_metrics.start_timer(Computation::SearchInit);

                ranges =
                    std::make_unique<std::pair<const uint32_t*, const uint32_t*>[]>(maps.size()+1);
                table_indices =
                    std::make_unique<uint_fast32_t[]>(maps.size()+1);

                query_objects.reserve(maps.size());
                std::transform(maps.begin(), maps.end(), std::back_inserter(query_objects),
                    [](auto& map) { return map.create_query(); });

                g_performance_metrics.store_time(Computation::SearchInit);
            }

            void fill_ranges(const std::vector<PrefixMap<THash>>& maps) {
                g_performance_metrics.start_timer(Computation::ReducePrefix);

                num_ranges = 0;
                for (uint_fast32_t j=0; j<maps.size(); j++) {
                    auto range = maps[j].get_next_range(query_objects[j]);
                    ranges[num_ranges] = range;
                    table_indices[num_ranges] = j;
                    // Skip empty ranges
                    num_ranges += (range.first != range.second);
                }
                // A large range that is never dereferenced, so that it will
                // never advance further in the array.
                ranges[num_ranges] = std::make_pair(&range_end_filler[0], &range_end_filler[2*RING_SIZE*4]);
                table_indices[num_ranges] = maps.size();

                g_performance_metrics.store_time(Computation::ReducePrefix);
            }
        };

        // Search the tables without any filters.
        void search_maps_no_filter(
            typename TSim::Format::Type* query,
            MaxBuffer& maxbuffer,
            float recall
        ) {
            SearchBuffers buffers(lsh_maps);
            for (uint_fast8_t depth=MAX_HASHBITS; depth > 0; depth--) {
                buffers.fill_ranges(lsh_maps);
                g_performance_metrics.start_timer(Computation::Consider);
                for (uint_fast32_t range_idx=0; range_idx < buffers.num_ranges; range_idx++) {
                    auto range = buffers.ranges[range_idx];
                    while (range.first != range.second) {
                        auto idx = *range.first;
                        auto dist = TSim::compute_similarity(
                            query,
                            dataset[idx],
                            dataset.get_description());
                        maxbuffer.insert(idx, dist);
                        range.first++;
                    }
                }
                g_performance_metrics.store_time(Computation::Consider);
                g_performance_metrics.start_timer(Computation::CheckTermination);
                auto kth_similarity = maxbuffer.smallest_value();
                auto table_idx = lsh_maps.size();
                auto last_tables = (depth == MAX_HASHBITS ? table_idx : lsh_maps.size());
                float failure_prob = hash_source->failure_probability(
                    depth,
                    table_idx,
                    last_tables,
                    kth_similarity
                );
                g_performance_metrics.store_time(Computation::CheckTermination);
                if (failure_prob <= 1-recall) {
                    g_performance_metrics.set_hash_length(depth);
                    g_performance_metrics.set_considered_maps(
                        (MAX_HASHBITS-depth+1)*lsh_maps.size());
                    return;
                }
            }
        }

        // Search maps with a simple implementation of filtering.
        void search_maps_simple_filter(
            typename TSim::Format::Type* query,
            MaxBuffer& maxbuffer,
            float recall
        ) {
            SearchBuffers buffers(lsh_maps);
            for (uint_fast8_t depth=MAX_HASHBITS; depth > 0; depth--) {
                buffers.fill_ranges(lsh_maps);
                g_performance_metrics.start_timer(Computation::Consider);
                for (uint_fast32_t range_idx=0; range_idx < buffers.num_ranges; range_idx++) {
                    auto range = buffers.ranges[range_idx];
                    while (range.first != range.second) {
                        auto idx = *range.first;
                        if (filterer.passes_filter(idx, range_idx%NUM_SKETCHES)) {
                            auto dist = TSim::compute_similarity(
                                query,
                                dataset[idx],
                                dataset.get_description());
                            maxbuffer.insert(idx, dist);
                        }
                        range.first++;
                    }
                    auto kth_similarity = maxbuffer.smallest_value();
                    filterer.update_max_sketch_diff(kth_similarity);
                }
                g_performance_metrics.store_time(Computation::Consider);
                g_performance_metrics.start_timer(Computation::CheckTermination);
                auto kth_similarity = maxbuffer.smallest_value();
                auto table_idx = lsh_maps.size();
                auto last_tables = (depth == MAX_HASHBITS ? table_idx : lsh_maps.size());
                float failure_prob = hash_source->failure_probability(
                    depth,
                    table_idx,
                    last_tables,
                    kth_similarity
                );
                g_performance_metrics.store_time(Computation::CheckTermination);
                if (failure_prob <= 1-recall) {
                    g_performance_metrics.set_hash_length(depth);
                    g_performance_metrics.set_considered_maps(
                        (MAX_HASHBITS-depth+1)*lsh_maps.size());
                    return;
                }
            }
        }

        // Search all maps and insert the candidates into the buffer.
        void search_maps(
            typename TSim::Format::Type* query,
            MaxBuffer& maxbuffer,
            float recall
        ) {
            const size_t FILTER_BUFFER_SIZE = 128;

            SearchBuffers buffers(lsh_maps);

            // Buffer for values passing filtering and should have distances computed.
            // 8*RING_SIZE is necessary additional space as that is the maximum that can be added
            // between the last check of the size and it being emptied.
            uint32_t passing_filter[FILTER_BUFFER_SIZE+8*RING_SIZE];

            // foreach possible bit in hash
            for (uint_fast8_t depth=MAX_HASHBITS; depth > 0; depth--) {
                // Find next ranges to consider
                buffers.fill_ranges(lsh_maps);
                g_performance_metrics.start_timer(Computation::Filtering);
                // Filter values
                const static int PREFETCH_DIST = 3;
                const static int PREREQ_PREFETCH_DIST = 5;
                const uint32_t* ring[RING_SIZE];
                // From which range are we currently moving values into the ring.
                uint_fast32_t range_idx = 0;

                // Number of values missing for the ring to be full
                // When there are missing values, the rest can contain invalid pointers that should not be dereferenced.
                // Prefetching is ok
                int_fast32_t missing_ring_vals = RING_SIZE;
                // Fill ring
                for (int_fast8_t i=0; i < RING_SIZE; i++) {
                    auto& range = buffers.ranges[range_idx];
                    ring[i] = range.first;
                    range.first += 4;
                    missing_ring_vals -= (range_idx < buffers.num_ranges);
                    range_idx += (range.first == range.second);
                }

                while (range_idx < buffers.num_ranges) {
                    uint_fast32_t num_passing_filter = 0;
                    // Can potentially add 4xRING_SIZE values to the buffer
                    while (num_passing_filter < FILTER_BUFFER_SIZE && missing_ring_vals == 0) {
                        // We know that the ring is full, so we can iter through it entirely.
                        // This should be completely unrolled
                        for (int_fast32_t ring_idx=0; ring_idx < RING_SIZE; ring_idx++) {
                            auto prefetch_ring_idx = (ring_idx+PREFETCH_DIST)&(RING_SIZE-1);
                            auto prefetch_segment = ring[prefetch_ring_idx];
                            filterer.prefetch(prefetch_segment[0], prefetch_ring_idx);
                            filterer.prefetch(prefetch_segment[1], prefetch_ring_idx);
                            filterer.prefetch(prefetch_segment[2], prefetch_ring_idx);
                            filterer.prefetch(prefetch_segment[3], prefetch_ring_idx);

                            auto prereq_prefetch_segment =
                                ring[(ring_idx+PREREQ_PREFETCH_DIST)&(RING_SIZE-1)];
                            __builtin_prefetch(&prereq_prefetch_segment[0]);
                            __builtin_prefetch(&prereq_prefetch_segment[1]);
                            __builtin_prefetch(&prereq_prefetch_segment[2]);
                            __builtin_prefetch(&prereq_prefetch_segment[3]);

                            auto v1 = ring[ring_idx][0];
                            auto v2 = ring[ring_idx][1];
                            auto v3 = ring[ring_idx][2];
                            auto v4 = ring[ring_idx][3];

                            auto p1 = filterer.passes_filter(v1, ring_idx);
                            auto p2 = filterer.passes_filter(v2, ring_idx);
                            auto p3 = filterer.passes_filter(v3, ring_idx);
                            auto p4 = filterer.passes_filter(v4, ring_idx);

                            passing_filter[num_passing_filter] = v1;
                            num_passing_filter += p1;
                            passing_filter[num_passing_filter] = v2;
                            num_passing_filter += p2;
                            passing_filter[num_passing_filter] = v3;
                            num_passing_filter += p3;
                            passing_filter[num_passing_filter] = v4;
                            num_passing_filter += p4;

                            // Put new query into the last slot
                            missing_ring_vals += (range_idx >= buffers.num_ranges);
                            auto& range = buffers.ranges[range_idx];
                            ring[ring_idx] = range.first;
                            range.first += 4;
                            range_idx += (range.first == range.second);
                        }
                        g_performance_metrics.add_candidates(RING_SIZE*4);
                    }
                    // Consider rest of values in ring when it isn't full.
                    // Can again add up to 4*RING_SIZE values to the buffer.
                    for (int_fast32_t ring_idx=RING_SIZE-1-missing_ring_vals; ring_idx >= 0; ring_idx--) {
                        auto prefetch_ring_idx = (ring_idx+PREFETCH_DIST)&(RING_SIZE-1);
                        auto prefetch_segment = ring[prefetch_ring_idx];

                        filterer.prefetch(prefetch_segment[0], prefetch_ring_idx);
                        filterer.prefetch(prefetch_segment[1], prefetch_ring_idx);
                        filterer.prefetch(prefetch_segment[2], prefetch_ring_idx);
                        filterer.prefetch(prefetch_segment[3], prefetch_ring_idx);

                        auto prereq_prefetch_segment =
                            ring[(ring_idx+PREREQ_PREFETCH_DIST)&(RING_SIZE-1)];
                        __builtin_prefetch(&prereq_prefetch_segment[0]);
                        __builtin_prefetch(&prereq_prefetch_segment[1]);
                        __builtin_prefetch(&prereq_prefetch_segment[2]);
                        __builtin_prefetch(&prereq_prefetch_segment[3]);

                        auto v1 = ring[ring_idx][0];
                        auto v2 = ring[ring_idx][1];
                        auto v3 = ring[ring_idx][2];
                        auto v4 = ring[ring_idx][3];

                        auto p1 = filterer.passes_filter(v1, ring_idx);
                        auto p2 = filterer.passes_filter(v2, ring_idx);
                        auto p3 = filterer.passes_filter(v3, ring_idx);
                        auto p4 = filterer.passes_filter(v4, ring_idx);

                        passing_filter[num_passing_filter] = v1;
                        num_passing_filter += p1;
                        passing_filter[num_passing_filter] = v2;
                        num_passing_filter += p2;
                        passing_filter[num_passing_filter] = v3;
                        num_passing_filter += p3;
                        passing_filter[num_passing_filter] = v4;
                        num_passing_filter += p4;
                    }
                    g_performance_metrics.add_candidates(4*(RING_SIZE-missing_ring_vals));

                    // Empty buffer
                    g_performance_metrics.store_time(Computation::Filtering);
                    g_performance_metrics.start_timer(Computation::Consider);
                    for (
                        uint_fast32_t passed_idx=0;
                        passed_idx < num_passing_filter;
                        passed_idx++
                    ) {
                        auto idx = passing_filter[passed_idx];
                        auto dist = TSim::compute_similarity(
                            query,
                            dataset[idx],
                            dataset.get_description());
                        maxbuffer.insert(idx, dist);
                    }
                    g_performance_metrics.add_distance_computations(num_passing_filter);
                    num_passing_filter = 0;
                    auto kth_similarity = maxbuffer.smallest_value();
                    filterer.update_max_sketch_diff(kth_similarity);
                    g_performance_metrics.store_time(Computation::Consider);

                    // Stop if we have seen enough to be confident about the recall guarantee
                    g_performance_metrics.start_timer(Computation::CheckTermination);
                    size_t table_idx = buffers.table_indices[range_idx];
                    auto last_tables = (depth == MAX_HASHBITS ? table_idx : lsh_maps.size());
                    float failure_prob = hash_source->failure_probability(
                        depth,
                        table_idx,
                        last_tables,
                        kth_similarity
                    );
                    g_performance_metrics.store_time(Computation::CheckTermination);
                    if (failure_prob <= 1-recall) {
                        g_performance_metrics.set_hash_length(depth);
                        g_performance_metrics.set_considered_maps(
                            (MAX_HASHBITS-depth)*lsh_maps.size()+table_idx);
                        return;
                    }
                    g_performance_metrics.start_timer(Computation::Filtering);
                }
                g_performance_metrics.store_time(Computation::Filtering);
            }
        }
    };

    template <typename T, typename U, typename V>
    constexpr IndependentHashArgs<U> Index<T, U, V>::DEFAULT_HASH_SOURCE;
    template <typename T, typename U, typename V>
    constexpr IndependentHashArgs<V> Index<T, U, V>::DEFAULT_SKETCH_SOURCE;
}
