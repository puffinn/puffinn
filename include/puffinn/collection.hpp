#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/filterer.hpp"
#include "puffinn/hash_source/deserialize.hpp"
#include "puffinn/hash_source/hash_source.hpp"
#include "puffinn/hash_source/independent.hpp"
#include "puffinn/maxbuffer.hpp"
#include "puffinn/maxpairbuffer.hpp"
#include "puffinn/prefixmap.hpp"
#include "puffinn/typedefs.hpp"

#include <cassert>
#include <istream>
#include <iostream>
#include <memory>
#include <ostream>
#include <unordered_set>
#include <vector>

// for debugging
#include <chrono>

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

    class ChunkSerializable {
    public:
        virtual void serialize_chunk(std::ostream&, size_t) const = 0;
    };

    /// Iterator over serialized chunks in the index.
    class SerializeIter {
        const ChunkSerializable& ref;
        size_t len;
        size_t idx = 0;

    public:
        SerializeIter(const ChunkSerializable& ref, size_t len)
          : ref(ref),
            len(len)
        {
        }

        bool has_next() const {
            return idx < len;
        }

        void serialize_next(std::ostream& out) {
            ref.serialize_chunk(out, idx);
            idx++;
        }
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
    class Index : ChunkSerializable {
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
            const HashSourceArgs<THash>& hash_args = IndependentHashArgs<THash>(),
            const HashSourceArgs<TSketch>& sketch_args = IndependentHashArgs<TSketch>()
        )
          : dataset(Dataset<typename TSim::Format>(dataset_args)),
            filterer(sketch_args, dataset.get_description()),
            memory_limit(memory_limit),
            hash_args(hash_args.copy())
        {
            static_assert(
                std::is_same<TSim, typename THash::Sim>::value
                && std::is_same<TSim, typename TSketch::Sim>::value,
                "Hash function not applicable to similarity measure");
        }

        /// Deserialize an index.
        ///
        /// It is assumed that the input data is a serialized index
        /// using the same version of PUFFINN.
        Index(std::istream& in)
          : dataset(in),
            filterer(in)
        {
            hash_args = deserialize_hash_args<THash>(in);
            bool has_hash_source;
            in.read(reinterpret_cast<char*>(&has_hash_source), sizeof(bool));
            if (has_hash_source) {
                hash_source = hash_args->deserialize_source(in);
            }
            size_t num_maps;
            in.read(reinterpret_cast<char*>(&num_maps), sizeof(size_t));
            lsh_maps.reserve(num_maps);
            bool use_chunks;
            in.read(reinterpret_cast<char*>(&use_chunks), sizeof(bool));
            if (!use_chunks) {
                for (size_t i=0; i < num_maps; i++) {
                    // if num_maps is non-zero, hash_source is non-null
                    lsh_maps.emplace_back(in, *hash_source);
                }
            }
            in.read(reinterpret_cast<char*>(&memory_limit), sizeof(uint64_t));
            in.read(reinterpret_cast<char*>(&last_rebuild), sizeof(uint32_t));
        }

        /// Deserialize a single chunk.
        void deserialize_chunk(std::istream& in) {
            // Assumes that hash_source is non-null,
            // which it will be if there were any chunks during serialization.
            lsh_maps.emplace_back(in, *hash_source);
        }

        /// Serialize the index to the output stream to be loaded later.
        /// Supports splitting the serialized data into chunks,
        /// which can be accessed using the ``serialize_chunks`` method.
        /// This is primarily useful when the serialized data cannot be written to a file directly
        /// and storing the index twice in memory is infeasible.
        ///
        /// @param use_chunks Whether to split the serialized index into chunks. Defaults to false.
        void serialize(std::ostream& out, bool use_chunks = false) const {
            dataset.serialize(out);
            filterer.serialize(out);
            hash_args->serialize(out);
            bool has_hash_source = hash_source.get() != nullptr;
            out.write(reinterpret_cast<char*>(&has_hash_source), sizeof(bool));
            if (has_hash_source) {
                hash_source->serialize(out);
            }
            size_t num_maps = lsh_maps.size();
            out.write(reinterpret_cast<char*>(&num_maps), sizeof(size_t));
            out.write(reinterpret_cast<char*>(&use_chunks), sizeof(bool));
            if (!use_chunks) {
                for (auto& m : lsh_maps) {
                    m.serialize(out);
                }
            }
            out.write(reinterpret_cast<const char*>(&memory_limit), sizeof(uint64_t));
            out.write(reinterpret_cast<const char*>(&last_rebuild), sizeof(uint32_t));
        }

        /// Get an iterator over serialized chunks in the dataset.
        /// See ``serialize`` for its use.
        SerializeIter serialize_chunks() const {
            return SerializeIter(*this, lsh_maps.size());
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

        /// Retrieve the n'th value inserted into the index.
        ///
        /// Since the value is converted back from the internal storage format,
        /// it is unlikely to be equal to the inserted value
        /// due to normalization, rounding and other adjustments.
        template <typename T>
        T get(uint32_t idx) {
            return convert_stored_type<typename TSim::Format, T>(
                dataset[idx],
                dataset.get_description());
        }

        /// Rebuild the index using the currently inserted points.
        /// 
        /// This is done in parallel by default.
        /// The number of threads used can be specified using the
        /// OMP_NUM_THREADS environment variable.
        void rebuild() {
            // Compute sketches for the new vectors.
            auto start_sketches = std::chrono::steady_clock::now();
            filterer.add_sketches(dataset, last_rebuild);
            auto end_sketches = std::chrono::steady_clock::now();
            auto elapsed_sketches = std::chrono::duration_cast<std::chrono::nanoseconds>(end_sketches - start_sketches).count();
            printf("Time to compute sketches %ld ns\n", elapsed_sketches);

            auto desc = dataset.get_description();
            auto table_bytes = PrefixMap<THash>::memory_usage(dataset.get_size(), hash_args->function_memory_usage(desc, MAX_HASHBITS));
            auto filterer_bytes = filterer.memory_usage(desc);

            uint64_t required_mem = dataset.memory_usage()+filterer_bytes; 
            unsigned int num_tables = 0;
            uint64_t table_mem = 0;
            while (required_mem + table_mem < memory_limit) {
                num_tables++;
                table_mem = hash_args->memory_usage(desc, num_tables, MAX_HASHBITS)
                    + num_tables * table_bytes;
            }
            if (num_tables != 0) {
                num_tables--;
            }

            // Not enough memory for at least one table
            if (num_tables == 0) {
                throw std::invalid_argument("insufficient memory");
            }

            printf("Building %d tables\n", num_tables);

            auto start = std::chrono::steady_clock::now();
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
                lsh_maps.reserve(num_tables);
                for (unsigned int repetition=0; repetition < num_tables; repetition++) {
                    lsh_maps.emplace_back(this->hash_source->sample(), MAX_HASHBITS);
                }
            }
            auto sources_sampled = std::chrono::steady_clock::now();

            for (auto& map : lsh_maps) {
                map.reserve(dataset.get_size());
            }

            // Compute hashes for the new vectors in order, so that caching works.
            // Hash a vector in all the different ways needed.
            for (size_t idx=last_rebuild; idx < dataset.get_size(); idx++) {
                auto hash_state = this->hash_source->reset(dataset[idx], true);
                // Only parallelize if this step is computationally expensive.
                if (hash_source->precomputed_hashes()) {
                    for (auto& map : lsh_maps) {
                        map.insert(idx, hash_state.get());
                    }
                } else {
                    #pragma omp parallel for
                    for (size_t map_idx = 0; map_idx < lsh_maps.size(); map_idx++) {
                        lsh_maps[map_idx].insert(idx, hash_state.get());
                    }
                }
            }

            for (size_t map_idx = 0; map_idx < lsh_maps.size(); map_idx++) {
                lsh_maps[map_idx].rebuild();
            }
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            printf("Time to build the index %ld ns\n\tbuilding sources: %12ld ns\n\tcomputing hashes: %12ld ns\n", 
                elapsed,
                std::chrono::duration_cast<std::chrono::nanoseconds>(sources_sampled - start).count(),
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - sources_sampled).count()
            );
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
        ) const {
            auto desc = dataset.get_description();
            auto stored_query = to_stored_type<typename TSim::Format>(query, desc);
            return search_formatted_query(stored_query.get(), k, recall, filter_type);
        }

        /// Search for the approximate ``k`` nearest neighbors to a value already inserted into the index.
        ///
        /// This is similar to ``search(get(idx))``, but avoids potential rounding errors
        /// from converting between data formats and automatically removes the query index
        /// from the search results .
        std::vector<uint32_t> search_from_index(
            uint32_t idx,
            unsigned int k,
            float recall,
            FilterType filter_type = FilterType::Default
        ) const {
            // search for one more as the query will be part of the result set.
            auto res = search_formatted_query(dataset[idx], k+1, recall, filter_type);
            if (res.size() != 0 && res[0] == idx) {
                res.erase(res.begin());
            } else {
                res.pop_back();
            }
            return res;
        }

        /// Compute a bruteforce per-point top-K self-join on the current index.
        ///
        /// 
        std::vector<std::vector<uint32_t>> bf_join(
            unsigned int k,
            FilterType filter_type = FilterType::Default
        ) const {
            std::vector<std::vector<uint32_t>> res;
            for (size_t i = 0; i < dataset.get_size(); i++) {
                res.push_back(search_bf_formatted_query(dataset[i], k));
            }
            return res;
        }

        bool check_active_counts(const std::vector<uint32_t>& indices, const std::vector<uint32_t>& segments,
            const bool* active, const std::vector<uint32_t>& active_counts) {

            std::cout << active_counts.size() << std::endl;
            std::cout << segments.size() - 1 << std::endl;
            std::cout << segments[segments.size() - 1] << std::endl;
            for (uint32_t j = 2; j < segments.size(); j++) {
                auto left = segments[j - 1];
                auto right = segments[j] - 1;
                auto cnt = 0;

                std::cout << j << " (" << left << " --- " << right << ") of " << segments.size() - 1 << std::endl;
                for (size_t l = left; l <= right; l++) {
                    if (l <= 0 || l >= indices.size()) {
                        std::cout << l << std::endl;
                    }
                    if (active[indices[l]]) {
                        cnt++;
                        if (indices[l] == 0) {
                            cnt = active_counts[j - 1];
                            break;
                        }
                    }
                }
                if (cnt != active_counts[j-1]) {
                    std::cout << "error at " << j << ": " << cnt << " " << active_counts[j-1] << std::endl;
                    return false;
                }
            }
            std::cout << "Finished check!" << std::endl;
            return true;
        }
        
        /// Compute a per-point top-K self-join on the current index with ``recall``.
        ///
        /// This carries out an individual  query for each point in the index.
        std::vector<std::vector<uint32_t>> naive_lsh_join(
            unsigned int k,
            float recall,
            FilterType filter_type = FilterType::Default
        ) const {
            std::vector<std::vector<uint32_t>> res;
            for (size_t i = 0; i < dataset.get_size(); i++) {
                res.push_back(search_formatted_query(dataset[i], k, recall, filter_type));
            }
            return res;
        }

    public:
        /// Compute a per-point top-K self-join on the current index with ``recall``.
        ///
        /// 
        std::vector<std::pair<uint32_t, uint32_t>> global_lsh_join(
            unsigned int k,
            float recall,
            FilterType filter_type = FilterType::Default
        ) {
            g_performance_metrics.new_query();
            g_performance_metrics.start_timer(Computation::Total);
            
            // Allocate a buffer for each data point.
            auto maxbuffer = MaxPairBuffer(k);

            // Store segments efficiently (?).
            // indices in segments[i][j-1], ..., segments[i][j]-1 in lsh_maps[i]
            // share the same hash code.
            std::vector<std::vector<uint32_t>> segments (lsh_maps.size());

            g_performance_metrics.start_timer(Computation::SearchInit);

            // Set up data structures. Create segments for initial hash codes.
            for (size_t i = 0; i < lsh_maps.size(); i++) {
                segments[i].push_back(0);
                for (size_t j = 1; j < lsh_maps[i].hashes.size(); j++) {
                    if (lsh_maps[i].hashes[j] != lsh_maps[i].hashes[j-1]) {
                        segments[i].push_back(j);
                    }
                }                
                // Carry out initial all-to-all comparisons within a segment.
                // We leave out the first and last segment since it's filled up with filler elements.
                for (size_t j = 2; j < segments[i].size() - 1; j++) { 
                    auto range = lsh_maps[i].get_segment(segments[i][j-1], segments[i][j]);
                    for (auto r = range.first; r != range.second; r++) {
                        for (auto s = r + 1; s != range.second; s++) {
                            auto R = *r;
                            auto S = *s;
                            // std::cout << "Comparing " << R << " and " << S << std::endl;
                            // comparisons++;
                            auto dist = TSim::compute_similarity(
                                dataset[R], 
                                dataset[S], 
                                dataset.get_description());
                            maxbuffer.insert(std::make_pair(R, S), dist);
                        }
                    }
                }            
            }
            g_performance_metrics.store_time(Computation::SearchInit);
            
            uint32_t prefix_mask = 0xffffffff;
            for (int depth = MAX_HASHBITS; depth >= 0; depth--) {
                // check current level
                g_performance_metrics.start_timer(Computation::Search);
                std::cout << "Checking level " << depth << std::endl;
                std::vector<std::vector<uint32_t>> new_segments (lsh_maps.size());

                for (size_t i = 0; i < lsh_maps.size(); i++) {
                    new_segments[i].push_back(0);

                    // check each pair of adjacent segments in lsh_maps[i] in ``depth``.
                    for (size_t j = 2; j < segments[i].size() - 1; j++) {
                        auto left = (lsh_maps[i].hashes[segments[i][j - 1]]) & prefix_mask;
                        auto actual = (lsh_maps[i].hashes[segments[i][j]]) & prefix_mask;
                        if (left == actual) {
                            for (uint32_t r = segments[i][j-1]; r < segments[i][j]; r++) {
                                for (uint32_t s = segments[i][j]; s < segments[i][j + 1]; s++) {
                                    auto R = lsh_maps[i].indices[r];
                                    auto S = lsh_maps[i].indices[s];

                                    auto dist = TSim::compute_similarity(
                                        dataset[R], 
                                        dataset[S], 
                                        dataset.get_description());
                                    maxbuffer.insert(std::make_pair(R, S), dist);
                                }
                            }
                        } else {
                            new_segments[i].push_back(segments[i][j]);
                        }
                    }
                } 
                g_performance_metrics.store_time(Computation::Search);   

                std::cout << " Check termination." << std::endl;

                // remove inactive nodes
                auto kth_similarity = maxbuffer.smallest_value();
                auto table_idx = lsh_maps.size();
                auto last_tables = (depth == MAX_HASHBITS ? table_idx : lsh_maps.size());
                float failure_prob = hash_source->failure_probability(
                    depth,
                    table_idx,
                    last_tables,
                    kth_similarity
                );
                // g_performance_metrics.store_time(Computation::CheckTermination);
                if (failure_prob <= 1-recall) {
                    break;
                }

                // prepare next round
                segments = new_segments;
                prefix_mask <<= 1;
            }
            g_performance_metrics.store_time(Computation::Total);
            std::cout << k << "-th largest similarity: " << maxbuffer.smallest_value() << std::endl;
            return maxbuffer.best_indices();
        }

        std::vector<std::pair<uint32_t, uint32_t>> global_bf_join(unsigned int k) {
            MaxPairBuffer maxbuffer(k);
            g_performance_metrics.new_query();
            g_performance_metrics.start_timer(Computation::Total);
            for (size_t r = 0; r < dataset.get_size(); r++) {
                for (size_t s = r + 1; s < dataset.get_size(); s++) {
                    auto dist = TSim::compute_similarity(
                        dataset[r], 
                        dataset[s], 
                        dataset.get_description());
                    maxbuffer.insert(std::make_pair(r, s), dist);
                }
            }
            g_performance_metrics.store_time(Computation::Total);
            std::cout << k << "-th distance: " << maxbuffer.smallest_value() << std::endl;
            return maxbuffer.best_indices();
        }

        /// Compute a per-point top-K self-join on the current index with ``recall``.
        ///
        /// 
        std::vector<std::vector<uint32_t>> lsh_join(
            unsigned int k,
            float recall,
            FilterType filter_type = FilterType::Default
        ) {
            std::vector<std::vector<uint32_t>> res;

            g_performance_metrics.new_query();
            g_performance_metrics.start_timer(Computation::Total);
            
            // Allocate a buffer for each data point.
            std::vector<MaxBuffer> maxbuffers;

            // Is a point still active?
            bool active[dataset.get_size()];

            // The set of active points.
            std::unordered_set<uint32_t> active_nodes;
            
            for (size_t i = 0; i < dataset.get_size(); i++) {
                maxbuffers.push_back(MaxBuffer(k));
                active[i] = true;
                active_nodes.insert(i);
            }

            // Store segments efficiently (?).
            // indices in segments[i][j-1], ..., segments[i][j]-1 in lsh_maps[i]
            // share the same hash code.
            std::vector<std::vector<uint32_t>> segments (lsh_maps.size());
            // Store count of active nodes in each segment.
            // active_count[i][j] = number of active nodes in 
            // segments[i][j-1], ..., segments[i][j]-1.
            // std::vector<std::vector<uint32_t>> active_count (lsh_maps.size());

            // node_positions[i][j]: position of node i in lsh_maps[j]
            // std::vector<std::vector<uint32_t>> node_positions (dataset.get_size());


            // int comparisons = 0;
            g_performance_metrics.start_timer(Computation::SearchInit);


            // Set up data structures. Create segments for initial hash codes.
            for (size_t i = 0; i < lsh_maps.size(); i++) {
                segments[i].push_back(0);
                for (size_t j = 1; j < lsh_maps[i].hashes.size(); j++) {
                    if (lsh_maps[i].hashes[j] != lsh_maps[i].hashes[j-1]) {
                        segments[i].push_back(j);
                        //active_count[i].push_back(j - segments[segments.size() - 2]);
                    }
                    //node_positions[indices[j]].push_back(j);
                }    

                // Carry out initial all-to-all comparisons within a segment.
                // We leave out the first and last segment since it's filled up with filler elements.
                for (size_t j = 2; j < segments[i].size() - 1; j++) { 
                    auto range = lsh_maps[i].get_segment(segments[i][j-1], segments[i][j]);
                    for (auto r = range.first; r != range.second; r++) {
                        for (auto s = r; s != range.second; s++) {
                            auto R = *r;
                            auto S = *s;
                            auto dist = TSim::compute_similarity(
                                dataset[R], 
                                dataset[S], 
                                dataset.get_description());
                            maxbuffers[R].insert(S, dist);
                            maxbuffers[S].insert(R, dist);
                        }
                    }
                }



            }
            g_performance_metrics.store_time(Computation::SearchInit);
            std::cout << "Initial scan done" << std::endl;

            // std::cout << "Current segments: " << std::endl;
            // for (auto& s: segments) {
            //         std::cout << s << " ";
            // }
            // std::cout << std::endl;

            // for (auto& idx: lsh_maps[0].indices) {
            //      std::cout << idx << " ";
            // }
            // std::cout << std::endl;
            // for (auto& h: lsh_maps[0].hashes) {
            //     std::cout << h << " ";
            // }
            // std::cout << std::endl;

            
            uint32_t prefix_mask = 0xffffffff;
            for (int depth = MAX_HASHBITS; depth >= 0; depth--) {
                // check current level
                g_performance_metrics.start_timer(Computation::Search);
                std::cout << "Checking level " << depth << std::endl;
                std::cout << "Active nodes: " << active_nodes.size() << std::endl;
                if (active_nodes.size() == 0) {
                    break;
                }
                std::vector<std::vector<uint32_t>> new_segments (lsh_maps.size());
                // std::vector<std::vector<uint32_t>> new_active_count (lsh_maps.size());

                // std::cout << "Current segments: " << std::endl;
                // for (auto& s: segments[0]) {
                //      std::cout << s << " ";
                // }
                // std::cout << std::endl;

                for (size_t i = 0; i < lsh_maps.size(); i++) {
                    new_segments[i].push_back(0);

                    // check each pair of adjacent segments in lsh_maps[i] in ``depth``.
                    for (size_t j = 2; j < segments[i].size() - 1; j++) {
                        auto left = (lsh_maps[i].hashes[segments[i][j - 1]]) & prefix_mask;
                        auto actual = (lsh_maps[i].hashes[segments[i][j]]) & prefix_mask;
                        if (left == actual) {
                            // if (active_count[i][j - 1] == 0 && active_count[i][j] == 0) {
                            //     continue;
                            // }
                            for (auto r = segments[i][j-1]; r < segments[i][j]; r++) {
                                for (auto  s = segments[i][j]; s < segments[i][j + 1]; s++) {
                                    auto R = lsh_maps[i].indices[r];
                                    auto S = lsh_maps[i].indices[s];
                                    // std::cout << "Comparing " << R << " and " << S << std::endl;
                                    // comparisons++;
                                    if (!active[R] && !active[S]) {
                                         continue;
                                    }
                                    // auto sketch_idx = j%NUM_SKETCHES;
                                    // auto r_sketch = filterer.get_sketch(R, sketch_idx);
                                    // auto s_sketch = filterer.get_sketch(S, sketch_idx);
                                
                                    // if (filterer.passes_filter(r_sketch, s_sketch, filterer.get_max_sketch_diff(maxbuffers[R]->smallest_value()))) {
                                        auto dist = TSim::compute_similarity(
                                            dataset[R], 
                                            dataset[S], 
                                            dataset.get_description());
                                        maxbuffers[R].insert(S, dist);
                                        maxbuffers[S].insert(R, dist);
                                    // }
                                    // collisions[R]++;
                                    // collisions[S]++;
                                }
                            }
                            // active_count[i][j] += active_count[i][j-1];                            
                        } else {
                            new_segments[i].push_back(segments[i][j]);
                            // new_active_count[i].push_back(active_count[i][j-1]);

                        }
                    }
                }    
                g_performance_metrics.store_time(Computation::Search);   

                // std::cout << new_segments.size() << std::endl;
                // for (auto& h: lsh_maps[0].hashes) {
                //      std::cout << (h & prefix_mask) << " ";
                // }
                // std::cout << std::endl;
                
                // for (auto& s: segments) {
                //     std::cout << s << ";";
                // }
                // std::cout << std::endl;

                std::unordered_set<uint32_t> new_active (active_nodes.size());

                std::cout << " Removing inactive nodes." << std::endl;
                
                g_performance_metrics.start_timer(Computation::Filtering);
                // remove inactive nodes
                for (auto& v: active_nodes) {
                    auto kth_similarity = maxbuffers[v].smallest_value();
                    auto table_idx = lsh_maps.size();
                    auto last_tables = (depth == MAX_HASHBITS ? table_idx : lsh_maps.size());
                    float failure_prob = hash_source->failure_probability(
                        depth,
                        table_idx,
                        last_tables,
                        kth_similarity
                    );
                    // g_performance_metrics.store_time(Computation::CheckTermination);
                    if (failure_prob > 1-recall) {
                        // g_performance_metrics.set_hash_length(depth);
                        // g_performance_metrics.set_considered_maps(
                            // (MAX_HASHBITS-depth+1)*lsh_maps.size());
                        // return;
                        //std::cout << kth_similarity << std::endl;
                        new_active.insert(v);
                    } else {
                        //std::cout << failure_prob << std::endl;
                        active[v] = false;
                        // std::cout << "Removing " << v << std::endl;
                        // TODO: 0 seems to be used by the filler segments, but is also the first actual data point.
                        // if (v == 0)
                        //     continue;
                        // for (size_t i = 0; i < node_positions[v].size(); i++) {
                        //     auto pos = node_positions[v][i];
                        //     auto it = std::upper_bound(new_segments[i].begin(), new_segments[i].end(), pos);
                        //     auto j = (it - new_segments[i].begin());
                        //     new_active_count[i][j-1]--;
                        // }
                    }
                }
                g_performance_metrics.store_time(Computation::Filtering);

                // prepare next round
                segments = new_segments;
                active_nodes = new_active;
                // active_count = new_active_count;
                //check_active_counts(lsh_maps[0].indices, segments[0], active, active_count[0]);
                prefix_mask <<= 1;
            }
            // auto n = dataset.get_size();
            // std::cout << "comparisons: " << comparisons << "; should be " << (n * (n - 1) / 2 + n) << std::endl;
            g_performance_metrics.store_time(Computation::Total);

            for (size_t i = 0; i < dataset.get_size(); i++) {
                auto best = maxbuffers[i].best_indices();
                // if (best.size() != k) {
                    // std::cout << "error! " << best.size() << " " << collisions[i] << std::endl;
                // }
                res.push_back(best);
            }
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
            return search_bf_formatted_query(stored.get(), k);
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
        std::vector<unsigned int> search_bf_formatted_query(
            typename TSim::Format::Type* query,
            unsigned int k
        ) const {
            MaxBuffer res(k);
            for (size_t i=0; i < dataset.get_size(); i++) {
                float sim = TSim::compute_similarity(
                    query,
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

        std::vector<uint32_t> search_formatted_query(
            typename TSim::Format::Type* query,
            unsigned int k,
            float recall,
            FilterType filter_type
        ) const {
            if (dataset.get_size() < 100) {
                // Due to optimizations values near the edges in prefixmaps are discarded.
                // When there are fewer total values than SEGMENT_SIZE, all values will be skipped.
                // However at that point, brute force is likely to be faster regardless.
                return search_bf_formatted_query(query, k);
            }
            g_performance_metrics.new_query();
            g_performance_metrics.start_timer(Computation::Total);

            MaxBuffer maxbuffer(k);
            g_performance_metrics.start_timer(Computation::Hashing);
            auto hash_state = hash_source->reset(query, false);
            g_performance_metrics.store_time(Computation::Hashing);
            g_performance_metrics.start_timer(Computation::Sketching);
            auto sketches = filterer.reset(query);
            g_performance_metrics.store_time(Computation::Sketching);

            g_performance_metrics.start_timer(Computation::Search);
            switch (filter_type) {
                case FilterType::None:
                    search_maps_no_filter(
                        query,
                        maxbuffer,
                        recall,
                        sketches,
                        hash_state.get());
                    break;
                case FilterType::Simple:
                    search_maps_simple_filter(
                        query,
                        maxbuffer,
                        recall,
                        sketches,
                        hash_state.get());
                    break;
                default:
                    search_maps(query, maxbuffer, recall, sketches, hash_state.get());
            }
            g_performance_metrics.store_time(Computation::Search);

            auto res = maxbuffer.best_indices();
            g_performance_metrics.store_time(Computation::Total);
            return res;
        }

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

            QuerySketches sketches;

            SearchBuffers(
                const std::vector<PrefixMap<THash>>& maps,
                QuerySketches sketches,
                HashSourceState* hash_state
            )
              : sketches(sketches)
            {
                g_performance_metrics.start_timer(Computation::SearchInit);

                ranges =
                    std::make_unique<std::pair<const uint32_t*, const uint32_t*>[]>(maps.size()+1);
                table_indices =
                    std::make_unique<uint_fast32_t[]>(maps.size()+1);

                query_objects.reserve(maps.size());
                std::transform(maps.begin(), maps.end(), std::back_inserter(query_objects),
                    [hash_state](auto& map) { return map.create_query(hash_state); });

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
            float recall,
            QuerySketches sketches,
            HashSourceState* hash_state
        ) const {
            SearchBuffers buffers(lsh_maps, sketches, hash_state);
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
            float recall,
            QuerySketches sketches,
            HashSourceState* hash_state
        ) const {
            SearchBuffers buffers(lsh_maps, sketches, hash_state);
            for (uint_fast8_t depth=MAX_HASHBITS; depth > 0; depth--) {
                buffers.fill_ranges(lsh_maps);
                g_performance_metrics.start_timer(Computation::Consider);
                for (uint_fast32_t range_idx=0; range_idx < buffers.num_ranges; range_idx++) {
                    auto range = buffers.ranges[range_idx];
                    while (range.first != range.second) {
                        auto idx = *range.first;
                        auto sketch_idx = range_idx%NUM_SKETCHES;
                        auto sketch = filterer.get_sketch(idx, sketch_idx);
                        if (buffers.sketches.passes_filter(sketch, sketch_idx)) {
                            auto dist = TSim::compute_similarity(
                                query,
                                dataset[idx],
                                dataset.get_description());
                            maxbuffer.insert(idx, dist);
                        }
                        range.first++;
                    }
                    auto kth_similarity = maxbuffer.smallest_value();
                    buffers.sketches.max_sketch_diff = filterer.get_max_sketch_diff(kth_similarity);
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
            float recall,
            QuerySketches sketches,
            HashSourceState* hash_state
        ) const {
            const size_t FILTER_BUFFER_SIZE = 128;

            SearchBuffers buffers(lsh_maps, sketches, hash_state);
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
                            prefetch_addr(&prereq_prefetch_segment[0]);
                            prefetch_addr(&prereq_prefetch_segment[1]);
                            prefetch_addr(&prereq_prefetch_segment[2]);
                            prefetch_addr(&prereq_prefetch_segment[3]);

                            // indices
                            auto v1 = ring[ring_idx][0];
                            auto v2 = ring[ring_idx][1];
                            auto v3 = ring[ring_idx][2];
                            auto v4 = ring[ring_idx][3];

                            // sketches
                            auto s1 = filterer.get_sketch(v1, ring_idx);
                            auto s2 = filterer.get_sketch(v2, ring_idx);
                            auto s3 = filterer.get_sketch(v3, ring_idx);
                            auto s4 = filterer.get_sketch(v4, ring_idx);

                            // Whether they pass the filtering step
                            auto p1 = buffers.sketches.passes_filter(s1, ring_idx);
                            auto p2 = buffers.sketches.passes_filter(s2, ring_idx);
                            auto p3 = buffers.sketches.passes_filter(s3, ring_idx);
                            auto p4 = buffers.sketches.passes_filter(s4, ring_idx);

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
                        prefetch_addr(&prereq_prefetch_segment[0]);
                        prefetch_addr(&prereq_prefetch_segment[1]);
                        prefetch_addr(&prereq_prefetch_segment[2]);
                        prefetch_addr(&prereq_prefetch_segment[3]);

                        auto v1 = ring[ring_idx][0];
                        auto v2 = ring[ring_idx][1];
                        auto v3 = ring[ring_idx][2];
                        auto v4 = ring[ring_idx][3];

                        auto p1 = buffers.sketches.passes_filter(v1, ring_idx);
                        auto p2 = buffers.sketches.passes_filter(v2, ring_idx);
                        auto p3 = buffers.sketches.passes_filter(v3, ring_idx);
                        auto p4 = buffers.sketches.passes_filter(v4, ring_idx);

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
                    buffers.sketches.max_sketch_diff = filterer.get_max_sketch_diff(kth_similarity);
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

        void serialize_chunk(std::ostream& out, size_t idx) const {
            lsh_maps[idx].serialize(out);
        }
    };
}
