set dotenv-load

check:
  cmake --build build --config RelWithDebInfo --target PuffinnJoin
  time build/PuffinnJoin < instructions.txt > result.dsv

cache-misses exe:
  perf record --call-graph dwarf -e cache-misses -p $(pgrep {{exe}})

profile exec:
  perf record --call-graph dwarf -p $(pgrep {{exec}})

# install flamegraph
install-flamegraph:
  # Check and install rust
  cargo --version || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  # Check and install flamegraph
  flamegraph --version || cargo install flamegraph

test:
  cmake --build build --config RelWithDebInfo --target Test
  env OMP_NUM_THREADS=56 build/Test "Jaccard*"

bench:
  cmake --build build --config RelWithDebInfo --target Bench
  env OMP_NUM_THREADS=56 build/Bench /mnt/large_storage/topk-join/datasets/orkut.hdf5 # >> bench_results.txt

# open the sqlite result database
sqlite:
  sqlite3 $TOPK_DIR/join-results.db

run:
  cmake --build build --config RelWithDebInfo --target PuffinnJoin
  cmake --build build --config RelWithDebInfo --target LSBTree
  cmake --build build --config RelWithDebInfo --target XiaoEtAl
  env TOPK_DIR=/mnt/large_storage/topk-join/ python3 join-experiments/run.py

plot:
  env TOPK_DIR=/mnt/large_storage/topk-join/ python3 join-experiments/plot.py

console:
  cd join-experiments && env TOPK_DIR=/mnt/large_storage/topk-join/ python3

compute-recalls:
  cd join-experiments && env TOPK_DIR=/mnt/large_storage/topk-join/ python3 -c 'import run; run.compute_recalls(run.get_db())'

