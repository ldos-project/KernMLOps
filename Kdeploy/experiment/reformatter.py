import sys
import statistics
import math

with open(sys.argv[1]) as f:
    size = 0
    times = []
    for line in f.readlines():
        if "SIZE" not in line:
            time = int(line)
            #time = int(line.split(" ")[1])
            times.append(time)
        else:
            if times != []:
                assert(len(times) == 40)
                times = sorted(times)[5:-5]
                stderr = statistics.stdev(times) / math.sqrt(len(times))
                print("Size %d: time: %d, std error: %.3f" % (size, sum(times) // len(times), stderr))
                times = []
            size = int(line.split(" ")[2])

    assert(len(times) == 40)
    times = sorted(times)[5:-5]
    stderr = statistics.stdev(times) / math.sqrt(len(times))
    print("Size %d: time: %d, std error: %.3f" % (size, sum(times) // len(times), stderr))


