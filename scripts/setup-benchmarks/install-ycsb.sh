#Specify destination for benchmark
YCSB_BENCHMARK_NAME="ycsb"
BENCHMARK_DIR_NAME="kernmlops-benchmark"

BENCHMARK_DIR="${BENCHMARK_DIR:-$HOME/$BENCHMARK_DIR_NAME}"
YCSB_BENCHMARK_DIR="$BENCHMARK_DIR/$YCSB_BENCHMARK_NAME"

if [ -d $YCSB_BENCHMARK_DIR ]; then
    echo "Benchmark already installed at: $YCSB_BENCHMARK_DIR"
    exit 0
fi

# Setup
mkdir $YCSB_BENCHMARK_DIR

pushd $YCSB_BENCHMARK_DIR
curl -O --location https://github.com/brianfrankcooper/YCSB/releases/download/0.17.0/ycsb-0.17.0.tar.gz
tar xfvz ycsb-0.17.0.tar.gz
mv ycsb-0.17.0 YCSB

popd

# Copy contents of ycsb_runner.py to bin/ycsb
cp scripts/setup-benchmarks/ycsb_runner.py $YCSB_BENCHMARK_DIR/YCSB/bin/ycsb

# Make the ycsb script executable
chmod +x $YCSB_BENCHMARK_DIR/YCSB/bin/ycsb
