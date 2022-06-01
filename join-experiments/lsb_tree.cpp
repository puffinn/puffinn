// Implements Algorithm CP3 of the paper
//   Efficient and Accurate Nearest Neighbor and Closest Pair Search in High-Dimensional Space
//   Yufei Tao, Ke Yi, Cheng Sheng, Panos Kalnis
//   Transactions on Database Systems
//   https://dl.acm.org/doi/pdf/10.1145/1806907.1806912

#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <chrono>
#include <omp.h>
#include "protocol.hpp"
#include "puffinn.hpp"
#include "puffinn/performance.hpp"

struct Pair
{
  size_t a;
  size_t b;
  float distance;
};

// Returns true if pair a is < than pair b, i.e. if a.distance < b.distance
bool cmp_pairs(const Pair &a, const Pair &b)
{
  return a.distance < b.distance;
}

int64_t zorder(std::vector<uint64_t> & gridded, size_t nbits) {
  int64_t z = 0;
  size_t nwords = gridded.size();

  for (size_t i = 0; i < nbits; i++) {
    for (size_t j = 0; j < nwords; j++) {
      z |= (gridded[j] & (1 << i)) << (j * nwords + i);
    }
  }

  return z;
}

float compute_extent(std::vector<std::vector<float>> & dataset) {
  float extent = 0.0;
  for (auto & v : dataset) {
    float max = -std::numeric_limits<float>::infinity();
    float min = std::numeric_limits<float>::infinity();
    for (auto c : v) {
      if (c > max) { max = c; }
      if (c < min) { min = c; }
    }
    float diff = max - min;
    if (diff > extent) { extent = diff; }
  }
  return extent;
}

float dotp(std::vector<float> &a, std::vector<float> &b) {
  size_t d = a.size();
  float p = 0.0;
  for (size_t i=0; i<d; i++) {
    p += a[i] * b[i];
  }
  return p;
}

float euclidean(std::vector<float> &a, std::vector<float> &b) {
  size_t d = a.size();
  float s = 0.0;
  for (size_t i=0; i<d; i++) {
    float diff = (a[i] - b[i]);
    s += diff * diff;
  }
  return std::sqrt(s);
}

std::pair<std::vector<std::pair<uint64_t, size_t>>, size_t> build_index(std::vector<std::vector<float>> &dataset, size_t m, double w, size_t seed) {
  size_t n = dataset.size();
  size_t d = dataset[0].size();

  std::mt19937 rng;
  rng.seed(seed);

  float extent = compute_extent(dataset);
  float f = std::ceil( std::log2f(extent) + std::log2f(d) );

  // Instantiate hash functions
  std::uniform_real_distribution<> uniform(0.0, w * std::pow(2.0, f));
  std::normal_distribution<> normal{0,1};

  std::vector<std::vector<float>> a; // the random projection vectors, in row major order
  std::vector<float> b; // the random offsets

  for (size_t i=0; i<m; i++) {
    std::cerr << "random vector " << i << ": ";
    std::vector<float> v;
    for (size_t j=0; j<d; j++) {
      float x = normal(rng);
      v.push_back(x);
      std::cerr << x << " ";
    }
    std::cerr << std::endl;
    a.push_back(v);
    b.push_back(uniform(rng));
  }

  // project the vectors
  std::vector<float> projections;
  for (size_t i=0; i<n; i++) {
    std::vector<float> & v = dataset[i];
    for (size_t j=0; j<m; j++) {
      float p = dotp(v, a[j]);
      float proj = p + b[j];
      projections.push_back(proj);
    }
  }

  // Discretize onto the grid
  std::vector<float> mincoord;
  std::vector<float> maxcoord;
  for (size_t i=0; i<m; i++) {
    mincoord.push_back(std::numeric_limits<float>::infinity());
    maxcoord.push_back(-std::numeric_limits<float>::infinity());
  }
  for (size_t i=0; i<projections.size(); i++) {
    size_t coord = i % m;
    if (projections[i] < mincoord[coord]) {
      mincoord[coord] = projections[i];
    }
    if (projections[i] > maxcoord[coord]) {
      maxcoord[coord] = projections[i];
    }
  }
  // the number of bits needed to represent each coordinate, discretized 
  // in a grid of step w. In the paper this is called u
  size_t coord_grid_bits = 0;
  for (size_t i=0; i<m; i++) {
    float e = maxcoord[i] - mincoord[i];
    uint64_t ncells = (uint64_t) std::ceil(e / w);
    std::cerr << "extent of projected coordinate " << i 
              << " is " << e << " (" << mincoord[i] << " to " << maxcoord[i] << ")"
              << ", divided in " 
              << ncells << " blocks" << std::endl;
    if (ncells == 0) throw std::logic_error("coordinate with 0 blocks!");
    uint64_t bits = 1;
    while (ncells > (1 << bits)) { bits++; }
    if (bits > coord_grid_bits) {
      coord_grid_bits = bits;
    }
  }
  std::cerr << "we need " << coord_grid_bits << " bits to represent each gridded coordinate" << std::endl;

  assert(m * coord_grid_bits <= 64);

  std::vector<uint64_t> gridded;
  std::vector<std::pair<uint64_t, size_t>> index;
  for (size_t i=0; i<n; i++) {
    gridded.clear();
    for (size_t coord = i*m; coord < (i+1)*m; coord++) {
      float proj = projections[coord];
      uint64_t h = (uint64_t) std::floor((proj - mincoord[coord % m]) / w);
      gridded.push_back(h);
    }
    uint64_t z = zorder(gridded, coord_grid_bits);
    index.push_back(std::make_pair(z, i));
  }

  std::sort(index.begin(), index.end());

  return std::make_pair(index, coord_grid_bits);
}

// longest common prefix
size_t lcp(uint64_t a, uint64_t b) {
  size_t l = 0;
  while ( (a >> l) != (b >> l) ) {
    l--;
  }
  return 64 - l;
}

std::pair<size_t, size_t> find_segment(std::vector<std::pair<uint64_t, size_t>> & index, size_t from) {
  size_t to = from+1;
  while (to < index.size() && index[to].first == index[from].first) {
    to++;
  }
  return std::make_pair(from, to);
}

void merge(std::vector<Pair> & a, std::vector<Pair> & b, size_t k) {
  for (Pair p : b) {
    a.push_back(p);
    if (a.size() > k) {
      std::pop_heap(a.begin(), a.end(), cmp_pairs);
      a.pop_back();
    }
  }
}

std::vector<Pair> cp3(
  std::vector<std::vector<float>> &dataset, 
  std::vector<std::pair<uint64_t, size_t>> & index, 
  size_t k, 
  size_t m, 
  size_t coord_grid_bits)
{
  int nthreads = omp_get_max_threads();
  std::vector<std::vector<Pair>> tl_result(nthreads);

  std::pair<size_t, size_t> current = find_segment(index, 0);
  while (current.first < index.size()) {
    std::cerr << " current block from " << current.first << " to " << current.second << " with " << (current.second - current.first) << " elements" << std::endl;
    // Self join of the current node
    #pragma omp parallel for
    for (size_t i=current.first; i < current.second; i++) {
      int tid = omp_get_thread_num();
      auto & a = dataset[index[i].second];
      for (size_t j=i+1; j < current.second; j++) {
        auto & b = dataset[index[j].second];
        float d = euclidean(a, b);
        tl_result[tid].push_back(Pair{index[i].second, index[j].second, d});
        std::push_heap(tl_result[tid].begin(), tl_result[tid].end(), cmp_pairs);

        if (tl_result[tid].size() > k) {
          std::pop_heap(tl_result[tid].begin(), tl_result[tid].end(), cmp_pairs);
          tl_result[tid].pop_back();
        }
      }
    }
    for(int tid=1; tid < nthreads; tid++) {
      merge(tl_result[0], tl_result[tid], k);
    }

    // explore a few of the next nodes
    float best = std::numeric_limits<float>::infinity();
    std::pair<size_t, size_t> next = find_segment(index, current.second);
    while (next.first < index.size()) {
      // std::cerr << "   next block from " << next.first << " to " << next.second << " " << std::endl;
      // check if the next segment is too far away
      if (tl_result[0].size() == k) {
        float d = tl_result[0].front().distance;
        if (d < best) {
          best = d;
          std::cerr << " current best guess " << best << std::endl;
        }
      }
      float block_bound = std::pow(2.0, 1 + coord_grid_bits - std::floor(lcp(index[current.first].first, index[next.first].first) / m));
      if (best < block_bound) {
        std::cerr << " early stopping loop block bound=" << block_bound << std::endl;
        break;
      }
    
      #pragma omp parallel for
      for (size_t i=current.first; i < current.second; i++) {
        int tid = omp_get_thread_num();
        auto & a = dataset[index[i].second];
        for (size_t j=next.first; j < next.second; j++) {
          auto & b = dataset[index[j].second];

          float d = euclidean(a, b);
          tl_result[tid].push_back(Pair{index[i].second, index[j].second, d});
          std::push_heap(tl_result[tid].begin(), tl_result[tid].end(), cmp_pairs);

          if (tl_result[tid].size() > k) {
            std::pop_heap(tl_result[tid].begin(), tl_result[tid].end(), cmp_pairs);
            tl_result[tid].pop_back();
          }
        }
      }
      for(int tid=1; tid < nthreads; tid++) {
        merge(tl_result[0], tl_result[tid], k);
      }

      // go to the next block
      next = find_segment(index, next.second);
    }
    current = find_segment(index, current.second);
  }

  for(int tid=1; tid < nthreads; tid++) {
    merge(tl_result[0], tl_result[tid], k);
  }

  return tl_result[0];
}


int main(void) {
    // Read parameters
    std::string protocol_line;
    expect("setup");
    unsigned int k = 10;
    size_t m = 16;
    double w = 4.0;
    size_t seed = 1234;
    while (true) {
        std::getline(std::cin, protocol_line);
        if (protocol_line == "sppv1 end") {
            break;
        }
        std::istringstream line(protocol_line);
        std::string key;
        line >> key;
        if (key == "k") {
            line >> k;
        } else if (key == "m") {
            line >> m;
        } else if (key == "w") {
            line >> w;
        } else if (key == "seed") {
            line >> seed;
        } else {
          // ignore other parameters
        }
    }
    send("ok");

    // Read the dataset
    expect("data");
    expect("cosine");
    std::cerr << "[c++] receiving data" << std::endl;
    auto dataset = read_float_vectors_hdf5(true);
    std::cerr << "Loaded " << dataset.size() << " vectors of dimension " << dataset[0].size() << std::endl;
    send("ok");

    expect("index");
    auto index_pair = build_index(dataset, m, w, seed);
    send("ok");

    expect("workload");
    auto top_pairs = cp3(dataset, index_pair.first, k, m, index_pair.second);
    send("ok");

    bool check = false;
    std::vector<Pair> actual;

    if (check) {
      std::cerr << "checking actual" << std::endl;
      for (size_t i=0; i<dataset.size(); i++) {
        for (size_t j=i+1; j<dataset.size(); j++) {
          float d = euclidean(dataset[i], dataset[j]);
          actual.push_back(Pair{i, j, d});
          std::push_heap(actual.begin(), actual.end(), cmp_pairs);
          if (actual.size() > k) {
            std::pop_heap(actual.begin(), actual.end(), cmp_pairs);
            actual.pop_back();
          }
        }
      }
    }

    expect("result");
    // std::cerr << "[c++] results size " << res.size() << std::endl; 
    while (!top_pairs.empty()) {
      std::pop_heap(top_pairs.begin(), top_pairs.end(), cmp_pairs);
      auto p = top_pairs.back();
      top_pairs.pop_back();
      std::cout << p.a << " " << p.b << std::endl;
      if (check) {
        std::pop_heap(actual.begin(), actual.end(), cmp_pairs);
        auto pcheck = actual.back();
        actual.pop_back();
        std::cerr << "ground truth " << pcheck.a << " " << pcheck.b << " " << pcheck.distance << std::endl;
      }
    }
    send("end");

    return 0;
}

