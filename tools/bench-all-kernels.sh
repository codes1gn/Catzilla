
for kernel in {0..13}; do
    ./build/sgemm/sgemm -version ${kernel} -device 0 -repeat 100 -warmup 10 -size-m 4096 -size-n 4096 -size-k 4096 | tee "__bench_cache__/${kernel}_output.txt"
    sleep 1
done
