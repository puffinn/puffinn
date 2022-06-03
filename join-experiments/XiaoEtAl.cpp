// An implementation of the Top-k similarity join from the paper:
//
// C. Xiao, W. Wang, X. Lin and H. Shang,
// "Top-k Set Similarity Joins,"
// 2009 IEEE 25th International Conference on Data Engineering, 2009,
// pp. 916-927, doi: 10.1109/ICDE.2009.111.

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <chrono>
#include "protocol.hpp"
#include "puffinn.hpp"

struct Pair
{
  size_t a;
  size_t b;
  float similarity;
};

// Returns true if pair a is < than pair b, i.e. if a.similarity > b.similarity
bool cmp_pairs(const Pair &a, const Pair &b)
{
  return a.similarity > b.similarity;
}

struct Event
{
  size_t index;
  size_t prefix;
  size_t length;
  float similarity;
};

// Returns true if a < b
bool cmp_events(const Event &a, const Event &b)
{
  if (a.similarity == b.similarity) {
    return a.length < b.length;
  } else {
    return a.similarity < b.similarity;
  }
}

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator () (std::pair<T1, T2> const &pair) const
    {
        std::size_t h1 = std::hash<T1>()(pair.first);
        std::size_t h2 = std::hash<T2>()(pair.second);
 
        return h1 ^ h2;
    }
};

void cerr_vec(std::vector<uint32_t> & v) {
  std::cerr << "[";
  for (auto x : v) {
    std::cerr << " " << x;
  }
  std::cerr << "]\n";
}

float jaccard(const std::vector<uint32_t> * a, const std::vector<uint32_t> * b) {
    auto& lhs = *a;
    auto& rhs = *b;
    size_t a_len = lhs.size();
    size_t b_len = rhs.size();
    int intersection_size = 0;
    size_t lhs_idx = 0;
    size_t rhs_idx = 0;
    while (lhs_idx < a_len && rhs_idx < b_len) {
        if (lhs[lhs_idx] == rhs[rhs_idx]) {
            intersection_size++;
            lhs_idx++;
            rhs_idx++;
        } else if(lhs[lhs_idx] < rhs[rhs_idx]) {
            lhs_idx++;
        } else {
            rhs_idx++;
        }
    }
    float intersection = intersection_size;
    auto divisor = lhs.size()+rhs.size()-intersection;
    if (divisor == 0) {
        return 0;
    } else {
        return intersection/divisor;
    }
}

std::vector<Pair> topk(std::vector<std::vector<uint32_t>> &dataset, size_t universe, size_t k)
{
  size_t n = dataset.size();

  float sim_threshold = 0.0;

  // Initialize the output with k arbitrary pairs
  std::vector<Pair> output;
  output.reserve(k + 1);
  for (size_t i = 1; i < k+1; i++)
  {
    float s = jaccard(&dataset[0], &dataset[i]);
    output.push_back(Pair{0, i, s});
  }
  std::make_heap(output.begin(), output.end(), cmp_pairs);
  sim_threshold = output.front().similarity;

  // Initialize the events queue
  std::vector<Event> events;
  events.reserve(n);
  for (size_t i = 0; i < n; i++)
  {
    events.push_back(Event{i, 0, dataset[i].size(), 1.0});
  }
  std::make_heap(events.begin(), events.end(), cmp_events);

  // Initialize the inverted index
  std::vector<std::vector<size_t>> inverted_index(universe);

  // Initialize the set to hold already seen pairs
  std::unordered_set<std::pair<size_t, size_t>, pair_hash> already_seen;
  
  // Process all events
  while (!events.empty())
  {
    // Pop next event
    std::pop_heap(events.begin(), events.end(), cmp_events);
    Event e = events.back();
    events.pop_back();

    sim_threshold = output.front().similarity;
    // Check stopping condition
    if (e.similarity <= sim_threshold)
    {
      break; // End of the algorithm
    }

    std::vector<uint32_t> *x = &dataset[e.index];
    size_t size_x = x->size();
    uint32_t w = x->at(e.prefix);
    // Lookup in inverted index
    for (size_t idx : inverted_index.at(w))
    {
      std::vector<uint32_t> *y = &dataset[idx];
      size_t size_y = y->size();
      if (e.index != idx) {
        if (size_x * sim_threshold <= size_y && size_y <= size_x / sim_threshold)
        { // size filtering
          size_t 
              a = std::min(e.index, idx),
              b = std::max(e.index, idx);
          if (already_seen.count(std::make_pair(a,b)) == 0) {
            float s = jaccard(x, y);
            Pair p { a, b, s };
            // Push in the heap
            output.push_back(p);
            std::push_heap(output.begin(), output.end(), cmp_pairs);

            // Remove from the output the pair in excess, if any
            if (output.size() > k) {
              std::pop_heap(output.begin(), output.end(), cmp_pairs);
              output.back();
              output.pop_back();
            }

            sim_threshold = output.front().similarity;
            // Mark the pair as already seen
            already_seen.insert(std::make_pair(a, b));
          }
        }
      }
    }

    // Index the current prefix
    inverted_index[w].push_back(e.index);

    if (e.prefix + 1 < x-> size()) {
      // Reschedule the set for the next prefix
      float s = 1.0 - ((float)e.prefix) / x->size(); // 1 - (prefix + 1 - 1) / |x|
      Event new_event{
          e.index,
          e.prefix + 1,
          e.length,
          s};
      events.push_back(new_event);
      std::push_heap(events.begin(), events.end(), cmp_events);
    }
  }

  return output;
}

puffinn::Dataset<puffinn::SetFormat> read_sets(const std::string &filename)
{
  std::ifstream file(filename);

  if (!file.is_open())
  {
    throw std::invalid_argument("File not found");
  }

  std::string full_line;
  std::getline(file, full_line);
  std::istringstream line(full_line);

  size_t universe;
  line >> universe;
  std::cerr << "Universe size " << universe << std::endl;

  puffinn::Dataset<puffinn::SetFormat> dataset(universe);

  while (!file.eof())
  {
    full_line.clear();
    std::getline(file, full_line);
    std::istringstream line(full_line);

    std::vector<uint32_t> set;
    uint32_t val;
    while (line >> val)
    {
      set.push_back(val);
    }

    if (set.size() != 0)
    {
      dataset.insert(set);
    }
  }

  return dataset;
}

std::vector<Pair> all_2_all(const puffinn::Dataset<puffinn::SetFormat> &dataset, size_t k) {
  std::vector<Pair> out;
  const size_t n = dataset.get_size();
  for (size_t i = 0; i < n; i++) {
    auto x = dataset[i];
    for (size_t j = i + 1; j < n; j++) {
      auto y = dataset[j];
      float s = jaccard(x, y);
      out.push_back(Pair{i, j, s});
      std::push_heap(out.begin(), out.end(), cmp_pairs);

      if (out.size() > k) {
        std::pop_heap(out.begin(), out.end(), cmp_pairs);
        out.pop_back();
      }
    }
  }

  return out;
}

int main(void) {
    // Read the dataset
    expect("data");
    expect("jaccard");
    // std::cerr << "[c++] receiving data" << std::endl;
    auto dataset = read_int_vectors_hdf5();
    size_t universe = 0;
    for (auto & v : dataset) {
      for (auto x : v) {
        if (x > universe) {
          universe = x;
        }
      }
    }
    universe++;
    std::cerr << "Loaded " << dataset.size() 
              << " vectors from a universe of size " << universe << std::endl;
    send("ok");
        

    expect("index");
    // No index is built
    send("ok");

    while (true) {
      std::string next_workload = protocol_read();
      std::cerr << "received " << next_workload << std::endl;
      if (next_workload == "end_workloads") {
          break;
      }
      std::string workload_params_str = next_workload.substr(std::string("workload ").size());
      std::cerr << "NEW WORKLOAD ON INDEX " << workload_params_str << std::endl;

      // query params
      unsigned int k = 1;

      std::istringstream workload_params_stream(workload_params_str);
      while (true) {
          std::string key;
          workload_params_stream >> key;
          if (key == "") {
              break;
          }
          if (key == "k") {
              workload_params_stream >> k;
          } else {
              std::cout << "sppv1 err unknown parameter " << key << std::endl;
              throw std::invalid_argument("unknown parameter");
          }
      }

      auto top_pairs = topk(dataset, universe, k);
      send("ok");

      expect("result");
      // std::cerr << "[c++] results size " << res.size() << std::endl; 
      while (!top_pairs.empty()) {
        std::pop_heap(top_pairs.begin(), top_pairs.end(), cmp_pairs);
        auto p = top_pairs.back();
        top_pairs.pop_back();
        std::cout << p.a << " " << p.b << std::endl;
      }
      send("end");
    }


    return 0;
}