#ifndef _SET_GET_H_
#define _SET_GET_H_
#include <linux/types.h>

#define NAME "set_get"

enum GET_SET_COMMAND {
	GET_ONE = 0x10,
	GET_MAPPED = 0x100,
};

struct get_set_args {
	__u32 key;
	__u64 value;
	__u64 map_name;
};

#endif //_SET_GET_H_
