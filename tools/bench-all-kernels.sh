
for kernel in {0..12}; do
    echo ""
    ./build/sgemm/sgemm $kernel | tee "__bench_cache__/${kernel}_output.txt"
    sleep 2
done
