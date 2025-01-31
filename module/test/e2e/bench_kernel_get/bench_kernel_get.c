/*
 * get_set.h - Kernel module for testing get ops inside kernel from fstore
 */

#include <linux/module.h>	/* Needed by all modules */
#include <linux/printk.h>	/* Needed for pr_info() */
#include <linux/fs.h> 		/* Needed for ioctl api */
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/kdev_t.h>
#include <linux/hashtable.h>
#include <linux/vmalloc.h>
#include "../../../fstore/fstore.h"
#include "bench_kernel_get.h"

#define DEBUG 1

#ifndef BENCH_GET_DATA_SIZE
#define BENCH_GET_DATA_SIZE 8
#endif

#ifndef BENCH_GET_ARRAY_SIZE
#define BENCH_GET_ARRAY_SIZE 10
#endif

dev_t dev = 0;
static struct class *dev_class;
static struct cdev bench_get_many_cdev;

int fstore_get(u64 map_name,
		void* key,
		size_t key_size,
		void* value,
		size_t value_size);

int fstore_get_value_size(u64 map_name,
		size_t* size);

int fstore_get_num_keys(u64 map_name,
		size_t* size);

static long get_set_ioctl(struct file *file,
				unsigned int cmd,
				unsigned long data);

static struct file_operations fops = {
	.owner = THIS_MODULE,
	.read = NULL,
	.write = NULL,
	.open = NULL,
	.unlocked_ioctl = get_set_ioctl,
	.release = NULL,
};

typedef struct bench_get_args gsa_t;

typedef struct ShiftXor shift_xor;

__u32 returner;

const shift_xor START_RANDOM = {1, 4, 7, 13};

typedef struct data {
	__u32 size[BENCH_GET_DATA_SIZE/4];
} data_t;

inline void inline_memcpy(void* dest, const void* src, size_t count) {
	char* tmp = dest;
	const char *s = src;
	while(count--) *tmp++ = *s++;
}

data_t temp_buffer;

static int bench_get_many_array(data_t* buffer, __u64 times, __u64* nanos) {
	shift_xor rand = START_RANDOM;
	const size_t map_size = BENCH_GET_ARRAY_SIZE;
	const size_t data_size = BENCH_GET_DATA_SIZE;
	for(__u64 i = 0; i < map_size; i++)
		for(__u64 j = 0; j < data_size/sizeof(__u32); j++)
			buffer[i].size[j] = simplerand(&rand);

	rand = START_RANDOM;
	__u32 accumulator = 0;
	__u64 start = ktime_get_raw_fast_ns();
	for(__u64 i = 0; i < times; i++) {
		__u32 key = simplerand(&rand) % map_size;
		inline_memcpy(&temp_buffer, &buffer[key], data_size);
		for(__u32 j = 0; j < data_size/sizeof(__u32); j++)
		{
			accumulator ^= temp_buffer.size[j];
		}
	}
	__u64 stop = ktime_get_raw_fast_ns();
	*nanos = stop - start;
	returner ^= accumulator;
	return 0;
}

static int bench_get_many_map(__u64 map_name, __u64 times, __u64* nanos) {
	int err = 0;
	size_t size;
	err = fstore_get_value_size(map_name, &size);
	if(err != 0 || size != BENCH_GET_DATA_SIZE) {
		pr_err("%s:%d: Getting value size not working\n",
			__FILE__, __LINE__);
		return err;
	}
	err = fstore_get_num_keys(map_name, &size);
	if(err != 0 || size != BENCH_GET_ARRAY_SIZE) {
		pr_err("%s:%d: Getting value size not working\n",
			__FILE__, __LINE__);
		return err;
	}
	shift_xor rand = START_RANDOM;
	const size_t map_size = BENCH_GET_ARRAY_SIZE;
	const size_t data_size = BENCH_GET_DATA_SIZE;
	__u32 accumulator = 0;
	__u64 start = ktime_get_raw_fast_ns();
	for(__u64 i = 0; i < times; i++) {
		__u32 key = simplerand(&rand) % map_size;
		if(( err =
			fstore_get(map_name,
				&key, 4, &temp_buffer, data_size) )) {
			pr_err("%s:%d Huge error occurred fstore_get",
					__FILE__, __LINE__);
			goto cleanup;
		}
		for(__u32 j = 0; j < size/4; j++)
		{
			accumulator ^= temp_buffer.size[j];
		}
	}
	__u64 stop = ktime_get_raw_fast_ns();
	*nanos = stop - start;
	returner ^= accumulator;
cleanup:
	return err;
}

static long get_set_ioctl(struct file* file,
				unsigned int cmd,
				unsigned long data)
{
	int err = -EINVAL;
	gsa_t* uptr = (gsa_t*) data;
	gsa_t gsa;
	data_t* array = NULL;
	size_t alloc_size;
	if( copy_from_user(&gsa, (gsa_t*) data, sizeof(gsa_t)) )
	{
		pr_err("Getting initial struct impossible\n");
		err = -EINVAL;
		return err;
	}
	switch (cmd) {
	case BENCH_GET_MANY:
		err = bench_get_many_map(gsa.map_name, gsa.number, &gsa.number);
		break;
	case BENCH_GET_ARRAY:
		alloc_size = BENCH_GET_ARRAY_SIZE * BENCH_GET_DATA_SIZE;
		array = vmalloc(alloc_size);
		if( array == NULL ) {
			pr_info("%s:%d Out of memory for: %lu\n",
					__FILE__, __LINE__, alloc_size);
			err = -ENOMEM;
			break;
		}
		err = bench_get_many_array(array, gsa.number, &gsa.number);
		break;
	default:
		pr_info("%s:%d Invalid Command arrived %u\n",
				__FILE__, __LINE__, cmd);
		err = -EINVAL;
		break;
	}

	if( err == 0 && copy_to_user(&uptr->number,
				&(gsa.number),
				sizeof(__u64)) ) {
		pr_err("Copy to User was thwarted\n");
		err = -EINVAL;
	}

	if( array != NULL ) vfree(array);
	return err;
}

int __init init_module(void)
{
	/*Allocating Major number*/
	if((alloc_chrdev_region(&dev, 0, 1, NAME"_dev")) < 0){
		pr_err("Cannot allocate major number\n");
		return -1;
	}

	pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

	/*Creating cdev structure*/
	cdev_init(&bench_get_many_cdev, &fops);

	/*Adding character device to the system*/
	if((cdev_add(&bench_get_many_cdev, dev, 1)) < 0){
		pr_err("Cannot add the device to the system\n");
		goto r_class;
	}

	/*Creating struct class*/
	if(IS_ERR(dev_class = class_create(NAME "_class"))){
		pr_err("Cannot create the struct class\n");
		goto r_class;
	}

	/*Creating device*/
	if(IS_ERR(device_create(dev_class, NULL, dev, NULL, NAME "_device"))){
		pr_err("Cannot create the Device 1\n");
		goto r_device;
	}

	pr_info(NAME " Driver Insert...Done!!!\n");
	return 0;

r_device:
	class_destroy(dev_class);
r_class:
	unregister_chrdev_region(dev,1);
	return -1;
}

void __exit cleanup_module(void)
{
	/* release device*/
	device_destroy(dev_class,dev);
	class_destroy(dev_class);
	cdev_del(&bench_get_many_cdev);
	unregister_chrdev_region(dev, 1);

	pr_info(NAME " exit; returner: %u \n", returner);
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Aditya Tewari <adityaatewari@gmail.com>");
MODULE_DESCRIPTION("benchmark getting many in kmods for feature store");
