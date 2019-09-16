#include <puffinn.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace puffinn {
namespace python {

namespace py = pybind11;

struct PySerializeIter {
    SerializeIter iter;

    PySerializeIter(SerializeIter iter)
      : iter(iter)
    {
    }

    py::bytes next() {
        std::stringstream s(std::ios_base::out | std::ios_base::binary);
        if (iter.has_next()) {
            iter.serialize_next(s);
        } else {
            throw py::stop_iteration();
        }
        return py::bytes(s.str());
    }
};

struct AbstractIndex {
    virtual void rebuild() = 0;
    virtual std::vector<uint32_t> search_from_index(
        uint32_t idx,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) = 0;
    virtual void serialize(std::ostream& out) = 0;
    virtual std::string metric() = 0;
    virtual std::string hash_function() = 0;
    virtual PySerializeIter serialize_chunks() = 0;
    virtual void append_chunk(std::string) = 0;
};

// Interface for datasets of vectors of real numbers.
// Used to have the python interface be a list of floats.
class RealIndex : public AbstractIndex {
public:
    virtual void insert(const std::vector<float>& vec) = 0;
    virtual std::vector<float> get(uint32_t) = 0;
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
    AngularIndex(std::istream& stream) : table(stream)
    {
    }

    AngularIndex(unsigned int dimensions, uint64_t memory_limit, const HashSourceArgs<T>& hash_args)
      : table(dimensions, memory_limit, hash_args)
    {
    }

    void insert(const std::vector<float>& vec) {
        table.insert(vec);
    }

    std::vector<float> get(uint32_t idx) {
        return table.template get<std::vector<float>>(idx);
    }

    void rebuild() {
        table.rebuild();
    }

    std::vector<uint32_t> search(
        const std::vector<float>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) {
        return table.search(vec, k, recall, filter_type);
    }

    std::vector<uint32_t> search_from_index(
        uint32_t idx,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) {
        return table.search_from_index(idx, k, recall, filter_type);
    }

    void serialize(std::ostream& out) {
        table.serialize(out, true);
    }

    std::string metric() {
        return "angular";
    }

    std::string hash_function() {
        if (std::is_same<T, SimHash>::value) {
            return "simhash";
        } else if (std::is_same<T, CrossPolytopeHash>::value) {
            return "crosspolytope";
        } else if (std::is_same<T, FHTCrossPolytopeHash>::value) {
            return "fht_crosspolytope";
        }
        return "";
    }

    PySerializeIter serialize_chunks() {
        return PySerializeIter(table.serialize_chunks());
    }

    void append_chunk(std::string s) {
        std::stringstream stream(s);
        table.deserialize_chunk(stream);
    }
};

class AbstractSetIndex : public AbstractIndex {
public:
    virtual void insert(const std::vector<uint32_t>& vec) = 0;
    virtual std::vector<uint32_t> get(uint32_t idx) = 0;
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
    SetIndex(std::istream& stream) : table(stream)
    {
    }

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

    std::vector<uint32_t> get(uint32_t idx) {
        return table.template get<std::vector<uint32_t>>(idx);
    }

    void rebuild() {
        table.rebuild();
    }

    std::vector<uint32_t> search(
        const std::vector<uint32_t>& vec,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) {
        return table.search(vec, k, recall, filter_type);
    }

    std::vector<uint32_t> search_from_index(
        uint32_t idx,
        unsigned int k,
        float recall,
        FilterType filter_type
    ) {
        return table.search_from_index(idx, k, recall, filter_type);
    }

    void serialize(std::ostream& out) {
        table.serialize(out, true);
    }

    std::string metric() {
        return "jaccard";
    }

    std::string hash_function() {
        if (std::is_same<T, MinHash>::value) {
            return "minhash";
        } else if (std::is_same<T, MinHash1Bit>::value) {
            return "1bit_minhash";
        }
        return "";
    }

    PySerializeIter serialize_chunks() {
        return PySerializeIter(table.serialize_chunks());
    }

    void append_chunk(std::string s) {
        std::stringstream stream(s);
        table.deserialize_chunk(stream);
    }
};

class Index {
    std::unique_ptr<RealIndex> real_table;
    std::unique_ptr<AbstractSetIndex> set_table;
public:
    // Needed for pickling
    Index() {}

    Index(
        std::string metric,
        unsigned int dimensions,
        uint64_t memory_limit,
        const py::kwargs& kwargs
    ) {
        if (metric == "angular") {
            init_angular(dimensions, memory_limit, kwargs);
        } else if (metric == "jaccard") {
            init_jaccard(dimensions, memory_limit, kwargs);
        } else {
            throw std::invalid_argument("metric");
        }
    }

    void setstate(std::string metric, std::string hash_function, std::string data) {
        std::stringstream stream(data, std::ios_base::in | std::ios_base::binary);
        if (metric == "angular") {
            if (hash_function == "simhash") {
                real_table = std::make_unique<AngularIndex<SimHash>>(stream);
            } else if (hash_function == "crosspolytope") {
                real_table = std::make_unique<AngularIndex<CrossPolytopeHash>>(stream);
            } else if (hash_function == "fht_crosspolytope") {
                real_table = std::make_unique<AngularIndex<FHTCrossPolytopeHash>>(stream);
            } else {
                throw std::invalid_argument("hash_function");
            }
        } else if (metric == "jaccard") {
            if (hash_function == "minhash") {
                set_table = std::make_unique<SetIndex<MinHash>>(stream);
            } else if (hash_function == "1bit_minhash") {
                set_table = std::make_unique<SetIndex<MinHash1Bit>>(stream);
            } else {
                throw std::invalid_argument("hash_function");
            }
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

    py::object get(uint32_t idx) {
        if (real_table) {
            return py::cast(real_table->get(idx));
        } else {
            return py::cast(set_table->get(idx));
        }
    }

    void rebuild() {
        if (real_table) {
            real_table->rebuild();
        } else {
            set_table->rebuild();
        }
    }

    FilterType get_filter_type(const std::string& name) {
        FilterType filter_type;
        if (name == "default") {
            filter_type = FilterType::Default;
        } else if (name == "none") {
            filter_type = FilterType::None;
        } else if (name == "simple") {
            filter_type = FilterType::Simple;
        } else {
            throw std::invalid_argument("filter_type");
        }
        return filter_type;
    }

    std::vector<uint32_t> search(
        py::list list,
        unsigned int k,
        float recall,
        std::string filter_name
    ) {
        auto filter_type = get_filter_type(filter_name);
        if (real_table) {
            auto vec = list.cast<std::vector<float>>();
            return real_table->search(vec, k, recall, filter_type);
        } else {
            auto vec = list.cast<std::vector<unsigned int>>();
            return set_table->search(vec, k, recall, filter_type);
        }
    }

    std::vector<uint32_t> search_from_index(
        uint32_t idx,
        unsigned int k,
        float recall,
        std::string filter_name
    ) {
        auto filter_type = get_filter_type(filter_name);
        if (real_table) {
            return real_table->search_from_index(idx, k, recall, filter_type);
        } else {
            return set_table->search_from_index(idx, k, recall, filter_type);
        }
    }

    py::tuple reduce();

    py::bytes serialize() {
        std::stringstream s(std::ios_base::out | std::ios_base::binary);
        if (real_table) {
            real_table->serialize(s);
        } else if (set_table) {
            set_table->serialize(s);
        }
        return py::bytes(s.str());
    }

    PySerializeIter serialize_chunks() {
        if (real_table) {
            return real_table->serialize_chunks();
        } else if (set_table) {
            return set_table->serialize_chunks();
        }
        throw std::exception();
    }

    void append_chunk(std::string s) {
        if (real_table) {
            real_table->append_chunk(s);
        } else if (set_table) {
            set_table->append_chunk(s);
        }
    }

    void extend_chunks(py::sequence seq) {
        py::iterator iter = py::iter(seq);
        while (iter != py::iterator::sentinel()) {
            append_chunk((*iter).cast<std::string>());
            iter++;
        }
    }

    std::string metric() {
        if (real_table) {
            return real_table->metric();
        } else if (set_table) {
            return set_table->metric();
        }
        return "";
    }

    std::string hash_function() {
        if (real_table) {
            return real_table->hash_function();
        } else if (set_table) {
            return set_table->hash_function();
        }
        return "";
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

    void set_hash_args(MinHash::Args& args, const py::dict& params) {
        set(args.randomized_bits, params, "randomized_bits");
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

struct IndexConstructor {
    Index operator()(std::string metric, std::string hash_function, std::string data) {
        Index index;
        index.setstate(metric, hash_function, data);
        return index;
    };
};

py::tuple Index::reduce() {
    return py::make_tuple(
        IndexConstructor{},
        py::make_tuple(metric(), hash_function(), serialize()),
        py::none(),
        serialize_chunks()
    );
}

PYBIND11_MODULE(puffinn, m) {
    py::class_<Index>(m, "Index")
        .def(py::init<const std::string&, const unsigned int&, const uint64_t&, const py::kwargs&>())
        .def("insert", &Index::insert)
        .def("rebuild", &Index::rebuild)
        .def("search", &Index::search,
             py::arg("vec"), py::arg("k"), py::arg("recall"),
             py::arg("filter_type") = "default"
         )
        .def("search_from_index", &Index::search_from_index,
            py::arg("index"), py::arg("k"), py::arg("recall"),
            py::arg("filter_type") = "default"
        )
        .def("get", &Index::get)
        .def("__reduce__", &Index::reduce)
        .def("append", &Index::append_chunk)
        .def("extend", &Index::extend_chunks);

    py::class_<IndexConstructor>(m, "IndexConstructor")
        .def("__call__", &IndexConstructor::operator())
        .def("__getstate__", [](const IndexConstructor &p) {
            return py::make_tuple();
        })
        .def("__setstate__", [](IndexConstructor &p, py::tuple t) {
            new (&p) IndexConstructor();
        });

    py::class_<PySerializeIter>(m, "PySerializeIter")
        .def("__next__", &PySerializeIter::next);
}
} // namespace python
} // namespace puffinn
