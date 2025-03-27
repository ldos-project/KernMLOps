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

# Clone
pushd $YCSB_BENCHMARK_DIR
git clone https://github.com/tewaro/YCSB.git -b tewaro/quickfix-coreworkload-deletes-master --depth=1

# Build
pushd YCSB
mvn -pl site.ycsb:mongodb-binding -am clean package
mvn -pl site.ycsb:redis-binding -am clean package
mvn -pl site.ycsb:memcached-binding -am clean package

popd
popd

pwd

# Copy contents of ycsb_runner.py to bin/ycsb
cp scripts/setup-benchmarks/ycsb_runner.py $YCSB_BENCHMARK_DIR/YCSB/bin/ycsb

# Make the ycsb script executable
chmod +x $YCSB_BENCHMARK_DIR/YCSB/bin/ycsb
