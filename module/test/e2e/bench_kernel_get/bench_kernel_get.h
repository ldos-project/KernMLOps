#ifndef _SET_GET_H_
#define _SET_GET_H_
#include <linux/types.h>

#define NAME "bench_kernel_get"

enum GET_SET_COMMAND {
	BENCH_GET_MANY = 0x100,
	BENCH_GET_ARRAY = 0x10,
};

struct bench_get_args {
	__u64 map_name;
	__u64 number;
};

struct ShiftXor {
	__u64 w;
	__u64 x;
	__u64 y;
	__u64 z;
};

inline __u64 simplerand(struct ShiftXor* rand) {
	__u64 t = rand->x;
	t ^= t << 11;
	t ^= t >> 8;
	rand->x = rand->y;
	rand->y = rand->z;
	rand->z = rand->w;
	rand->w ^= rand->w >> 19;
	rand->w ^= t;
	return rand->w;
}

#endif //_SET_GET_H_
