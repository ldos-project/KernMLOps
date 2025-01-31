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
#include "../../../fstore/fstore.h"
#include "set_get.h"


dev_t dev = 0;
static struct class *dev_class;
static struct cdev set_get_cdev;

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

static long get_set_ioctl(struct file* file,
		unsigned int cmd,
		unsigned long data);

int fstore_get(u64 map_name,
		void* key,
		size_t key_size,
		void* value,
		size_t value_size);

int fstore_get_value_size(u64 map_name,
			size_t* size);

int fstore_get_num_keys(u64 map_name,
			size_t* size);

typedef struct get_set_args gsa_t;

static long get_set_ioctl(struct file* file,
				unsigned int cmd,
				unsigned long data)
{
	int err = -EINVAL;
	gsa_t* uptr = (gsa_t*) data;
	gsa_t gsa;
	size_t size = 0;
	switch (cmd) {
	case GET_ONE:
		if( copy_from_user(&gsa, (gsa_t*) data, sizeof(gsa_t)) )
		{
			pr_err("Getting initial struct impossible\n");
			err = -EINVAL;
			break;
		}
		err = fstore_get_value_size(gsa.map_name, &size);
		if( err != 0 ) {
			return err;
		} else if( size != 8 ) {
			return -EMSGSIZE;
		}
		err = fstore_get_num_keys(gsa.map_name, &size);
		if( err != 0 ) {
			return err;
		} else if( size != 100) {
			return -ENOMEM;
		}
		err = fstore_get(gsa.map_name,
				&gsa.key, sizeof(gsa.key),
				&gsa.value, sizeof(gsa.value));
		if(err == 0) {
			if( copy_to_user(&uptr->value,
					&(gsa.value),
					sizeof(gsa.value)) ) {
				pr_err("Returning was thwarted\n");
				err = -EINVAL;
			}
		}
	default:
		break;
	}

	return err;
}

int __init init_module(void)
{
	/*Allocating Major number*/
	if((alloc_chrdev_region(&dev, 0, 1, NAME"_dev")) <0){
					pr_err("Cannot allocate major number\n");
					return -1;
	}

	pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

	/*Creating cdev structure*/
	cdev_init(&set_get_cdev, &fops);

	/*Adding character device to the system*/
	if((cdev_add(&set_get_cdev, dev, 1)) < 0){
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
	cdev_del(&set_get_cdev);
	unregister_chrdev_region(dev, 1);

	pr_info(NAME " exit.\n");
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Aditya Tewari <adityaatewari@gmail.com>");
MODULE_DESCRIPTION("get map in userspace feature store");
