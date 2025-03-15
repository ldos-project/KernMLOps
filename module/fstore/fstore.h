#ifndef _FSTORE_H_
#define _FSTORE_H_
#include <asm-generic/int-ll64.h>

enum fstore_cmd {
	REGISTER_MAP = 0x0,
	UNREGISTER_MAP = 0x1,
};

struct register_input {
	__u64 map_name;
	__u32 fd;
};

#ifdef __cplusplus
#include <optional>

std::optional<__u64> convert8byteStringHash(char* string) {
	__u64 hash = 0;
	__u8 i = 0;
	for (; string[i] != '\0' && i < 8; i++) {
		hash |= ((__u64)string[i]) << (i * 8);
	}
	if (string[i] != '\0') return std::nullopt;
	return hash;
}

consteval __u64 unsafeHashConvert(const char* string) {
	__u64 hash = 0;
	__u8 i = 0;
	for (; string[i] != '\0' && i < 8; i++) {
		hash |= ((__u64)string[i]) << (i * 8);
	}
  	return hash;
}

#endif // __cplusplus

#endif // _FSTORE_H_
