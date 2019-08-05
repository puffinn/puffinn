#include <puffinn.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace puffinn {
namespace python {

namespace py = pybind11;

// Interface for datasets of vectors of real numbers.
// Used to have the python interface be a list of floats.
class RealIndex {
public:
    virtual void insert(const std::vector<float>& vec) = 0;
    virtual void rebuild(unsigned int num_threads) = 0;
    virtual std::vector<uint32_t> search(
        const std::vector<float>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) = 0;
};

template <typename T, typename U = SimHash>
class AngularIndex : public RealIndex {
    Index<CosineSimilarity, T, U> table;

public:
    AngularIndex(unsigned int dimensions, uint64_t memory_limit, const HashSourceArgs<T>& hash_args)
      : table(dimensions, memory_limit, hash_args)
    {
    }

    void insert(const std::vector<float>& vec) {
        table.insert(vec);
    };
    void rebuild(unsigned int num_threads) {
        table.rebuild(num_threads);
    }
    std::vector<uint32_t> search(
        const std::vector<float>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) {
        return table.search(vec, k, recall, filter_type);
    }
};

template <typename T, typename U = SimHash>
class EuclideanIndex : public RealIndex {
    Index<L2Distance, T, U> table;

public:
    EuclideanIndex(unsigned int dimensions, uint64_t memory_limit, const HashSourceArgs<T>& hash_args)
      : table(dimensions, memory_limit, hash_args)
    {
    }

    void insert(const std::vector<float>& vec) {
        table.insert(vec);
    };
    void rebuild(unsigned int num_threads) {
        table.rebuild(num_threads);
    }
    std::vector<uint32_t> search(
        const std::vector<float>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) {
        return table.search(vec, k, recall, filter_type);
    }
};

class AbstractSetIndex {
public:
    virtual void insert(const std::vector<uint32_t>& vec) = 0;
    virtual void rebuild(unsigned int num_threads) = 0;
    virtual std::vector<uint32_t> search(
        const std::vector<uint32_t>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) = 0;
};

template <typename T, typename U = MinHash1Bit>
class SetIndex : public AbstractSetIndex {
    Index<JaccardSimilarity, T, U> table;

public:
    SetIndex(
        unsigned int dimensions,
        uint64_t memory_limit,
        const HashSourceArgs<T>& hash_args
    ) 
      : table(dimensions, memory_limit, hash_args) 
    {
    }

    void insert(const std::vector<uint32_t>& vec) {
        table.insert(vec); 
    }

    void rebuild(unsigned int num_threads) {
        table.rebuild(num_threads);
    }

    std::vector<uint32_t> search(
        const std::vector<uint32_t>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) {
        return table.search(vec, k, recall, filter_type);
    }
};

class Index {
    std::unique_ptr<RealIndex> real_table;
    std::unique_ptr<AbstractSetIndex> set_table;
public:
    Index(
        std::string metric,
        unsigned int dimensions,
        uint64_t memory_limit,
        const py::kwargs& kwargs
    ) {
        if (metric == "angular") {
            init_angular(dimensions, memory_limit, kwargs);
        } else if (metric == "euclidean") {
            init_euclidean(dimensions, memory_limit, kwargs);
        } else if (metric == "jaccard") {
            init_jaccard(dimensions, memory_limit, kwargs);
        } else {
            throw std::invalid_argument("metric");
        }
    }

    void insert(py::list list) {
        if (real_table) {
            auto vec = list.cast<std::vector<float>>();
            real_table->insert(vec);
        } else {
            auto vec = list.cast<std::vector<unsigned int>>();
            set_table->insert(vec);
        }
    }

    void rebuild(unsigned int num_threads) {
        if (real_table) {
            real_table->rebuild(num_threads);
        } else {
            set_table->rebuild(num_threads);
        }
    }

    std::vector<uint32_t> search(
        py::list list,
        unsigned int k,
        float recall,
        std::string filter_name
    ) {
        FilterType filter_type;
        if (filter_name == "default") {
            filter_type = FilterType::Default;
        } else if (filter_name == "none") {
            filter_type = FilterType::None;
        } else if (filter_name == "simple") {
            filter_type = FilterType::Simple;
        } else {
            throw std::invalid_argument("filter_type");
        }
        if (real_table) {
            auto vec = list.cast<std::vector<float>>();
            return real_table->search(vec, k, recall, filter_type);
        } else {
            auto vec = list.cast<std::vector<unsigned int>>();
            return set_table->search(vec, k, recall, filter_type);
        }
    }

private:
    template <typename T>
    void set(T& field, const py::dict& params, const char* name) {
        if (params.contains(name)) {
            field = py::cast<T>(params[name]);
        }
    }

    // No args
    void set_hash_args(SimHash::Args&, const py::dict&) {
    }

    void set_hash_args(CrossPolytopeHash::Args& args, const py::dict& params) {
        set(args.estimation_repetitions, params, "estimation_repetitions");
        set(args.estimation_eps, params, "estimation_eps");
    }

    void set_hash_args(FHTCrossPolytopeHash::Args& args, const py::dict& params) {
        set(args.estimation_eps, params, "estimation_eps");
        set(args.estimation_repetitions, params, "estimation_repetitions");
        set(args.num_rotations, params, "num_rotations");
    }

    void set_hash_args(MinHash::Args& args, const py::dict&) {
        set(args.randomize_tokens, params, "randomize_tokens");
    }

    template <typename T>
    std::unique_ptr<HashSourceArgs<T>> get_hash_source_args(const py::kwargs& kwargs) {
        std::string source = "independent";
        if (kwargs.contains("hash_source")) {
            source = py::cast<std::string>(kwargs["hash_source"]);
        }

        if (source == "pool") {
            unsigned int pool_size = 3000; 
            if (kwargs.contains("source_args") && kwargs["source_args"].contains("pool_size")) {
                pool_size = py::cast<unsigned int>(kwargs["source_args"]["pool_size"]);
            }
            auto res = std::make_unique<HashPoolArgs<T>>(pool_size);
            if (kwargs.contains("hash_args")) {
                set_hash_args(res->args, kwargs["hash_args"]);
            }
            return res;
        } else if (source == "independent") {
            auto res = std::make_unique<IndependentHashArgs<T>>();
            if (kwargs.contains("hash_args")) {
                set_hash_args(res->args, kwargs["hash_args"]);
            }
            return res; 
        } else if (source == "tensor") {
            auto res = std::make_unique<TensoredHashArgs<T>>();
            if (kwargs.contains("hash_args")) {
                set_hash_args(res->args, kwargs["hash_args"]);
            }
            return res; 
        } else {
            throw std::invalid_argument("hash_source");
        }
    }

    void init_angular(unsigned int dimensions, uint64_t memory_limit, const py::kwargs& kwargs) {
        std::string hash_function = "fht_crosspolytope";
        if (kwargs.contains("hash_function")) {
            hash_function = py::cast<std::string>(kwargs["hash_function"]);
        }
        if (hash_function == "simhash") {
            real_table = std::make_unique<AngularIndex<SimHash>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<SimHash>(kwargs));
        } else if (hash_function == "crosspolytope") {
            real_table = std::make_unique<AngularIndex<CrossPolytopeHash>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<CrossPolytopeHash>(kwargs));
        } else if (hash_function == "fht_crosspolytope") {
            real_table = std::make_unique<AngularIndex<FHTCrossPolytopeHash>>(        
                dimensions,
                memory_limit,
                *get_hash_source_args<FHTCrossPolytopeHash>(kwargs));
        } else {
            throw std::invalid_argument("hash_function");
        }
    }

    void init_euclidean(unsigned int dimensions, uint64_t memory_limit, const py::kwargs& kwargs) {
        std::string hash_function = "fht_crosspolytope";
        if (kwargs.contains("hash_function")) {
            hash_function = py::cast<std::string>(kwargs["hash_function"]);
        }
        if (hash_function == "simhash") {
            real_table = std::make_unique<EuclideanIndex<SimHash>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<SimHash>(kwargs));
        } else if (hash_function == "crosspolytope") {
            real_table = std::make_unique<EuclideanIndex<CrossPolytopeHash>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<CrossPolytopeHash>(kwargs));
        } else if (hash_function == "fht_crosspolytope") {
            real_table = std::make_unique<EuclideanIndex<FHTCrossPolytopeHash>>(        
                dimensions,
                memory_limit,
                *get_hash_source_args<FHTCrossPolytopeHash>(kwargs));
        } else {
            throw std::invalid_argument("hash_function");
        }
    }

    void init_jaccard(unsigned int dimensions, uint64_t memory_limit, const py::kwargs& kwargs) {
        std::string hash_function = "minhash";
        if (kwargs.contains("hash_function")) {
            hash_function = py::cast<std::string>(kwargs["hash_function"]);
        }
        if (hash_function == "minhash") {
            set_table = std::make_unique<SetIndex<MinHash>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<MinHash>(kwargs));
        } else if (hash_function == "1bit_minhash") {
            set_table = std::make_unique<SetIndex<MinHash1Bit>>(
                dimensions,
                memory_limit,
                *get_hash_source_args<MinHash1Bit>(kwargs));
        } else {
            throw std::invalid_argument("hash_function");
        }
    }
};

PYBIND11_MODULE(puffinn, m) {
    py::class_<Index>(m, "Index")
        .def(py::init<const std::string&, const unsigned int&, const uint64_t&, const py::kwargs&>())
        .def("insert", &Index::insert)
        .def("rebuild", &Index::rebuild, py::arg("num_threads") = 0)
        .def("search", &Index::search,
             py::arg("vec"), py::arg("k"), py::arg("recall"),
             py::arg("filter_type") = "default"
         );
}
} // namespace python
} // namespace puffinn
