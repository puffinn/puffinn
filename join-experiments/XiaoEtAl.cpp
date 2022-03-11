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
  float similarity;
};

// Returns true if a < b
bool cmp_events(const Event &a, const Event &b)
{
  return a.similarity < b.similarity;
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

std::vector<Pair> topk(const puffinn::Dataset<puffinn::SetFormat> &dataset, size_t k)
{
  size_t n = dataset.get_size();
  size_t universe = dataset.get_description().args;
  auto descr = dataset.get_description();

  float sim_threshold = 0.0;

  // Initialize the output with k arbitrary pairs
  std::vector<Pair> output;
  output.reserve(k + 1);
  for (size_t i = 0; i < k; i++)
  {
    float s = puffinn::JaccardSimilarity::compute_similarity(dataset[i], dataset[i + 1], descr);
    output.push_back(Pair{i, i + 1, s});
    if (s > sim_threshold)
    {
      sim_threshold = s;
    }
  }
  std::make_heap(output.begin(), output.end(), cmp_pairs);
  printf("Similarity threshold %f\n", sim_threshold);

  // Initialize the events queue
  std::vector<Event> events;
  events.reserve(n);
  for (size_t i = 0; i < n; i++)
  {
    events.push_back(Event{i, 0, 1.0});
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
    printf("Similarity threshold %f, candidate upper bound %f, prefix %lld\n", sim_threshold, e.similarity, e.prefix);
    // Check stopping condition
    if (e.similarity <= sim_threshold)
    {
      break; // End of the algorithm
    }

    std::vector<uint32_t> *x = dataset[e.index];
    size_t size_x = x->size();
    uint32_t w = x->at(e.prefix);
    // printf("Looking for matches from %d at prefix length %d\n", e.index, w);
    // Lookup in inverted index
    for (size_t idx : inverted_index[w])
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
            float s = puffinn::JaccardSimilarity::compute_similarity(x, y, descr);
            Pair p { a, b, s };
            // Push in the heap
            output.push_back(p);
            std::push_heap(output.begin(), output.end(), cmp_pairs);

            // Remove from the output the pair in excess, if any
            if (output.size() > k) {
              std::pop_heap(output.begin(), output.end(), cmp_pairs);
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

    // Reschedule the set for the next prefix
    float s = 1.0 - ((float)e.prefix) / x->size(); // 1 - (prefix + 1 - 2) / |x|
    Event new_event{
        e.index,
        e.prefix + 1,
        s};
    events.push_back(new_event);
    std::push_heap(events.begin(), events.end(), cmp_events);
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
  std::cout << "Universe size " << universe << std::endl;

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

int main()
{
  auto dataset = read_sets("dblp.vecs.txt");
  std::cout << "Read " << dataset.get_size() << " sets" << std::endl;

  size_t k = 100;
  auto top_pairs = topk(dataset, k);
  while (!top_pairs.empty()) {
    std::pop_heap(top_pairs.begin(), top_pairs.end(), cmp_pairs);
    auto pair = top_pairs.back();
    top_pairs.pop_back();
    std::cout << pair.a << " " << pair.b << " @ " << pair.similarity << std::endl;
  }

  return 0;
}