# Feature Store Module

## Dependencies

```shell
sudo apt install flex bison pahole libssl-dev libelf-dev
```

## Setting up GRUB

Note: Fixing the `/etc/default/grub` should only need to be done once.
Then in `/etc/default/grub`,
please ensure you have it pointing to the *old* kernel.
My suggestion is to use a script like the one here:
[grub-menu.sh](https://askubuntu.com/questions/1019213/display-grub-menu-and-options-without-rebooting)

> Note that the boot menu was taken from the above link.

I suggest in `/etc/default/grub` changing the `GRUB_DEFAULT` variable
to be a long string version.

In the case with this menu:

```shell
0    Ubuntu                                                     ↑    │
        │     1    Advanced options for Ubuntu
        │     1>0  Ubuntu, with Linux 4.14.31-041431-generic
        │     1>1  Ubuntu, with Linux 4.14.31-041431-generic (upstart)
        │     1>2  Ubuntu, with Linux 4.14.31-041431-generic (recovery mode)
        │     1>3  Ubuntu, with Linux 4.14.30-041430-generic
        │     1>4  Ubuntu, with Linux 4.14.30-041430-generic (upstart)
        │     1>5  Ubuntu, with Linux 4.14.30-041430-generic (recovery mode)
        │     1>6  Ubuntu, with Linux 4.14.27-041427-generic
        │     1>7  Ubuntu, with Linux 4.14.27-041427-generic (upstart)
        │     1>8  Ubuntu, with Linux 4.14.27-041427-generic (recovery mode)
        │     1>9  Ubuntu, with Linux 4.14.24-041424-generic
        │     1>10 Ubuntu, with Linux 4.14.24-041424-generic (upstart)
        │     1>11 Ubuntu, with Linux 4.14.24-041424-generic (recovery mode)
        │     1>12 Ubuntu, with Linux 4.14.23-041423-generic
        │     1>13 Ubuntu, with Linux 4.14.23-041423-generic (upstart)
```

Something like:
`GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 4.14.31-041431-generic"`

### Booting Into the New Kernel

Then the following will help boot into the new kernel after the installation.

```shell
sudo grub-reboot "1>0" && sudo reboot
```

Where `"1>0"` is the short string corresponding to the kernel you want to boot into.

## Build and Deploy Kernel

This should mostly be handled by `make -C module -j$(nproc) ${MODULE}/lib`.
However first you need to introduce environment variables that are going to aid this.

```shell
export SUDO="sudo"
export MODULE=/
export KHEADS=/usr/src/linux-headers-6.12.0-gadc218676eef
export KBUILD=<some-build-src>
```

1. Build the Kernel
2. Build the Modules
3. Build the Headers
4. Install Headers
5. Install Modules
    - The above steps are handled by `make -C module all`.
6. Install Kernel
    - This is done with `sudo make -C ${KBUILD} install`

## Helpful tricks

### Repeated SSH

When working with ssh it can get pretty frustrating to wait for reboot.
I use the following script to simplify my life.

```shell
while ! ssh <ssh-args>; do sleep 1 && echo HI && sleep 1; done
```
