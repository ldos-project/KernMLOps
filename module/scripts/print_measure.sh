printf "Count, DataSize, MapSize, AverageTime, Variance\n"
for ds in 8 16 32 64 128 256 512; do
    for ms in 32 64 128 256 512 1024 2048; do
        FILENAME="$1-${ds}-${ms}.stats"
        #FILENAME="kmod-map-${ds}-${ms}.stats"
        #FILENAME="kmod-array-${ds}-${ms}.stats"
        STATS=$(awk '{print $8}' ${FILENAME} | stats-exe)
        COUNT=$(head -n 1 ${FILENAME} | awk '{print $2}')
        NONZERO=$(echo ${STATS} | awk '{print $1}')
        VARIANCE=$(echo ${STATS} | awk '{print $3}')
        printf "${COUNT}, ${ds}, ${ms}, ${NONZERO}, ${VARIANCE}\n"
    done
done
