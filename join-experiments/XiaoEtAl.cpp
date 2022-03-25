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

std::vector<Pair> topk(const puffinn::Dataset<puffinn::SetFormat> &dataset, size_t k)
{
  size_t n = dataset.get_size();
  size_t universe = dataset.get_description().args;

  float sim_threshold = 0.0;

  // Initialize the output with k arbitrary pairs
  std::vector<Pair> output;
  output.reserve(k + 1);
  for (size_t i = 1; i < k+1; i++)
  {
    float s = jaccard(dataset[0], dataset[i]);
    output.push_back(Pair{0, i, s});
  }
  std::make_heap(output.begin(), output.end(), cmp_pairs);
  sim_threshold = output.front().similarity;

  // Initialize the events queue
  std::vector<Event> events;
  events.reserve(n);
  for (size_t i = 0; i < n; i++)
  {
    events.push_back(Event{i, 0, dataset[i]->size(), 1.0});
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

    std::vector<uint32_t> *x = dataset[e.index];
    size_t size_x = x->size();
    uint32_t w = x->at(e.prefix);
    // Lookup in inverted index
    for (size_t idx : inverted_index.at(w))
    {
      std::vector<uint32_t> *y = dataset[idx];
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

int main(int argc, char **argv)
{
  if (argc < 3) {
    std::cerr << "USAGE: ./XiaoEtAl dataset topk (topk...)" << std::endl;
    return 1;
  }
  auto dataset = read_sets(argv[1]);
  std::cerr << "Read " << dataset.get_size() << " sets" << std::endl;

  size_t nruns = 1;

  std::cout << "| method | dataset |    k | elapsed (ms) | similarity |" << std::endl;
  std::cout << "| :----- | :------ | ---: | -----------: | ---------: |" << std::endl;
  for (int i = 2; i < argc; i++) {
    size_t k = atoi(argv[i]);
    auto top_pairs = topk(dataset, k);
    auto k_th_pair = top_pairs.front();

    // while (!top_pairs.empty()) {
    //   std::pop_heap(top_pairs.begin(), top_pairs.end(), cmp_pairs);
    //   auto p = top_pairs.back();
    //   top_pairs.pop_back();
    //   std::cerr << "  " << p.a << " " << p.b << " @ " << p.similarity << std::endl;
    // }
    // printf("\n");

    // take the average time
    auto start = std::chrono::steady_clock::now();
    for (size_t run=0; run < nruns; run++) {
      topk(dataset, k);
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / nruns;

    std::cout << "| Xiao et al. | " << argv[1] << " | " << k << " | " << elapsed_ms << " | " << k_th_pair.similarity << " |" << std::endl;

    // If the dataset is small enough, check for correctness using the all-2-all algorithm
    if (dataset.get_size() <= 10000) {
      printf("Exact all to all computation");
      auto check = all_2_all(dataset, k);

      while (!check.empty()) {
        std::pop_heap(check.begin(), check.end(), cmp_pairs);
        auto p = check.back();
        check.pop_back();
        std::cerr << "  " << p.a << " " << p.b << " @ " << p.similarity << std::endl;
      }
    }
  }
  
  auto universe_size = dataset.get_description().args;
  puffinn::Index<puffinn::JaccardSimilarity, puffinn::MinHash1Bit> index(
    universe_size,
    1e9,
    puffinn::TensoredHashArgs<puffinn::MinHash1Bit>());

  for (size_t i = 0; i < dataset.get_size(); i++) {
    index.insert(*dataset[i]);
  }
  index.rebuild();
  
  for (int i = 2; i < argc; i++) {
    size_t k = atoi(argv[i]);
    auto buffer = index.global_lsh_join(k, 0.8);  
    auto best_entries = buffer.best_entries();

    std::cout << "| LSHJoin | " << argv[1] << " | " << k << " | " << 
      puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Total) * 1000 << " | " <<
      buffer.smallest_value() << " | " << std::endl;
    // for (int j = k - 1; j >= 0; j--) {
    //   std::cout << best_entries[j].first.first << " " << best_entries[j].first.second << " @ " 
    //     << best_entries[j].second << " " << std::endl;
    // }
  }

  // for (int i = 2; i < argc; i++) {
  //   size_t k = atoi(argv[i]);
  //   auto buffer = index.global_bf_join(k);  
  //   auto best_entries = buffer.best_entries();
  //   std::cout << "| BF | " << argv[1] << " | " << k << " | " << 
  //     puffinn::g_performance_metrics.get_total_time(puffinn::Computation::Total) * 1000 << " | " <<
  //     buffer.smallest_value() << " | " << std::endl;
  // }


  return 0;
}