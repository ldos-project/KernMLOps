/*
 * fstore.c - Kernel module for exposing ebpf maps
 */

#include <linux/module.h>	/* Needed by all modules */
#include <linux/printk.h>	/* Needed for pr_info() */
#include <linux/fs.h> 		/* Needed for ioctl api */
#include <linux/bpf.h>		/* Needed to get the ebpf_map */
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/kdev_t.h>
#include <linux/hashtable.h>
#include <linux/types.h>
#include "fstore.h"

atomic64_t number_maps;

dev_t dev = 0;
static struct class *dev_class;
static struct cdev fstore_cdev;

static long fstore_ioctl(struct file *file,
				unsigned int cmd,
				unsigned long data);

int fstore_register(u32 fd, u64 map_name);
EXPORT_SYMBOL_GPL(fstore_register);
int fstore_unregister(u64 map_name);
EXPORT_SYMBOL_GPL(fstore_unregister);

struct file_operations fops = {
	.owner = THIS_MODULE,
	.read = NULL,
	.write = NULL,
	.open = NULL,
	.unlocked_ioctl = fstore_ioctl,
	.release = NULL,
};

#define FSTORE_LOGSIZE 8
#define FSTORE_SIZE (1 << FSTORE_LOGSIZE)
DEFINE_HASHTABLE(fstore_map, FSTORE_LOGSIZE);

typedef struct register_input register_t;

typedef struct hash_node
{
	struct bpf_map* map;
	struct hlist_node hnode;
} hash_t;

struct bpf_map* bpf_map_get(u32 ufd);

/**
 * register a map with the name given by a u64.
 */
int fstore_register(u32 fd, u64 map_name)
{
	int err = 0;
	/* acquire space to create map */
	u64 index = atomic64_inc_return_acquire(&number_maps);
	if(index >= FSTORE_SIZE) { err = -ENOSPC; goto cleanup_count; }

	/* Check if value exists in map */
	int i = 0;
	hash_t* item = NULL;
	hash_for_each_possible_rcu_notrace(fstore_map, item, hnode, map_name) {
		i++;
	}
	if(i >= 1) { err = -EEXIST; goto cleanup_count;}

	struct bpf_map* map;
	map = bpf_map_get(fd);
	if(IS_ERR(map)) {
		err = -EBADF;
		goto cleanup_count;
	}

	/* Check if name exists in map */
	item = kmalloc(sizeof(hash_t), GFP_KERNEL);
	if(!item) {
		err = -ENOMEM;
		goto cleanup_map;
	}

	item->map = map;
	hash_add_rcu(fstore_map, &item->hnode, map_name);
	return err;

cleanup_map:
	bpf_map_put(map);
cleanup_count:
	atomic64_dec_return_relaxed(&number_maps);
	return err;
}

static void fstore_delete(hash_t* item)
{
	hash_del_rcu(&item->hnode);
	atomic64_dec_return_release(&number_maps);
	struct bpf_map* map = item->map;
	if(IS_ERR(map)) bpf_map_put(map);
	kfree(item);
}

/**
 * fstore_unregister - unregisters a map name, deleting all related bpf maps
 * @map_name: the name of the map you want to delete u64ified
 * @ret returns the number of deleted map elements
 */
int fstore_unregister(u64 map_name) {
	int i = 0;
	hash_t* item = NULL;
	struct hlist_node* tmp;
	hash_for_each_possible_safe(fstore_map, item, tmp, hnode, map_name) {
		fstore_delete(item);
		i++;
	}

	return i;
}

/*
static int fstore_get_all(
		u64 map_name,
		char* buffer,
		size_t size,
		size_t* copied_size)
{
	return 0;
}
*/

static long fstore_ioctl(struct file *file,
				unsigned int cmd,
				unsigned long data)
{
	int err = 0;
	register_t input;
	switch(cmd) {
	case REGISTER_MAP:
		if( copy_from_user(&input,
			(register_t*) data,
			sizeof(register_t)) )
		{
			pr_err("Getting initial struct impossible\n");
			err = -EINVAL;
			break;
		}
		err = fstore_register(input.fd, input.map_name);
		break;

	case UNREGISTER_MAP:
		err = fstore_unregister((u64) data);
		break;

	default:
		pr_info("Default case");
	}
	return err;
}

/*
static int convertString2u64(const char* s, u64* ret) {
	*ret = 0;
	int i = 0;
	for(; i < 8 && s[i] != '\0'; i++) {
		*ret |= ((u64)s[i]) << (i*8);
	}
	if(s[i] != '\0') return -1;
	return 0;
}
*/

int __init init_module(void)
{
	atomic64_set(&number_maps, 0);

	/*Allocating Major number*/
	if((alloc_chrdev_region(&dev, 0, 1, "fstore_dev")) <0){
					pr_err("Cannot allocate major number\n");
					return -1;
	}

	pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

	/*Creating cdev structure*/
	cdev_init(&fstore_cdev,&fops);

	/*Adding character device to the system*/
	if((cdev_add(&fstore_cdev,dev,1)) < 0){
		pr_err("Cannot add the device to the system\n");
		goto r_class;
	}

	/*Creating struct class*/
	if(IS_ERR(dev_class = class_create("fstore_class"))){
		pr_err("Cannot create the struct class\n");
		goto r_class;
	}

	/*Creating device*/
	if(IS_ERR(device_create(dev_class,NULL,dev,NULL,"fstore_device"))){
		pr_err("Cannot create the Device 1\n");
		goto r_device;
	}

	hash_init(fstore_map);

	pr_info("Fstore Driver Insert...Done!!!\n");
	return 0;

r_device:
	class_destroy(dev_class);
r_class:
	unregister_chrdev_region(dev,1);
	return -1;
}

void	__exit cleanup_module(void)
{
	/* Clean up fstore_map*/
	int i;
	struct hlist_node* nodeptr;
	hash_t* ptr;
	hash_for_each_safe(fstore_map, i, nodeptr, ptr, hnode) {
		fstore_delete(ptr);
	}

	/* release device*/
	device_destroy(dev_class,dev);
	class_destroy(dev_class);
	cdev_del(&fstore_cdev);
	unregister_chrdev_region(dev, 1);

	pr_info("Fstore exit.\n");
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Aditya Tewari <adityaatewari@gmail.com>");
MODULE_DESCRIPTION("Simple feature store");
