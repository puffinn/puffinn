test:
        cmake --build build --config RelWithDebInfo --target Test
        env OMP_NUM_THREADS=56 build/Test

bench:
        cmake --build build --config RelWithDebInfo --target Bench
        env OMP_NUM_THREADS=56 build/Bench glove.25d.100k.txt >> bench_results.txt

run:
        cmake --build build --config RelWithDebInfo --target PuffinnJoin
        env TOPK_DIR=/mnt/large_storage/topk-join/ python3 join-experiments/run.py

plot:
        env TOPK_DIR=/mnt/large_storage/topk-join/ python3 join-experiments/plot.py
