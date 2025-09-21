import os

import torch
import torch._inductor.config as config


class Buffer:
    def __init__(self, name, shadow=True, shape=(), stride=()):
        self.name = name
        self.shadow = shadow
        self.shape = shape
        self.stride = stride
    def getShape(self, dim):
        if dim >= len(self.shape):
            return 1
        return self.shape[dim]
    def getStride(self, dim):
        if dim >= len(self.stride):
            return 1
        return self.stride[dim]

class TorchKernelDeployer:
    def __init__(self, model, input_shape):
        # TODO handle models with multiple inputs
        self.model = model

        self.input_shape = input_shape
        self.input_size_flat = 1
        for dim in self.input_shape:
            self.input_size_flat *= dim

        self.output_shape = None # will be updated when we compile the model
        self.output_size_flat = None

        self.buffers = {}

    @staticmethod
    def collect_parameters(module):
        '''Returns model weights in the order that the triton kernels expect them to be in'''
        params = []
        for name, p in module._parameters.items():
            if p is not None:
                params.append(p)
        for child in module.children():
            params.extend(TorchKernelDeployer.collect_parameters(child))
        return params

    def gen_primals_init_code(self, *inputs):
        '''
        Generate C code to initialize primals from a pytorch model
            model: Pytorch model to read primals from
            *inputs: Sample inputs to the model (need to explore multiple parameters)
        '''
        primals = []
        s = """
typedef struct {
    float** primals;
    int input_idx;
    int input_size;
} primals_t;

primals_t allocate_primals(void) {
"""
        primals = self.collect_parameters(self.model)
        insert_pos = 2 if len(primals) >= 2 else len(primals) # TODO hardcoded at 2 for now, not sure if this always works
        for i, inp in enumerate(inputs):
            # TODO currently assuming that multiple inputs are just inserted one after another, not sure if this is really the case
            primals.insert(insert_pos + i, inp)

        s += "\tprimals_t p;\n"
        s += f"\tp.input_idx = {insert_pos};\n"
        s += f"\tp.input_size = {len(primals[insert_pos])};\n"
        s += "\tp.primals = heap_alloc(%d * sizeof(float*));\n" % len(primals)
        for i in range(len(primals)):
            p = torch.flatten(primals[i]).tolist()
            s += "\tp.primals[%d] = heap_alloc(%d * sizeof(float));\n" % (i, len(p))
            for j in range(len(p)):
                s += "\tp.primals[%d][%d] = %.20f;\n" % (i, j, p[j])

        s += "\treturn p;\n}\n"

        s += f"""
void free_primals(primals_t p) {{
    for (int i = 0; i < {len(primals)}; i++) {{
        heap_free(p.primals[i]);
    }}
    heap_free(p.primals);
}}
"""

        return s

    def dump_torch_files(self):
        '''Tell pytorch to compile the model and dump triton kernels/python glue into a "dump" folder'''
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TORCH_LOGS"] = "output_code"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "dump"
        config.cpu_backend = "triton"

        compiled = torch.compile(self.model, fullgraph=True, backend="inductor")
        out = compiled(torch.randn(self.input_shape, dtype=torch.float32))
        self.output_shape = out.shape
        self.output_size_flat = 1
        for dim in self.output_shape:
            self.output_size_flat *= dim

    def save_triton_kernels(self, target_dir="build"):
        '''Move triton kernels (asm) from the dump folder to some target folder'''
        os.system(f"mkdir -p {target_dir}")
        os.system(f"mv dump/triton/*/*/*.asm {target_dir}")

    def get_python_triton_wrapper(self):
        '''Return the python wrapper that launches the triton kernels'''
        file_name = os.popen('grep -r "def call(args):" dump').read().split(":")[0]
        with open(file_name) as file:
            contents = file.readlines()
            start = contents.index("def call(args):\n")
            end = start
            while contents[end] != "\n":
                end += 1

            # dont include the indentation
            return [contents[i][4:] for i in range(start + 1, end)]

    @staticmethod
    def get_torch_type_size(s):
        '''
        Get the size in bytes of a torch type

        Ex: get_torch_type_size("torch.float32") -> 32
        '''
        return eval(s).itemsize

    @staticmethod
    def get_c_type(triton_type):
        '''
        Get the C type for a given triton type

        Ex: get_c_type("f32") -> "float"
        '''
        if triton_type[0] == "*":
            return "void*"
        elif triton_type[0] == "f":
            # TODO handle different sized floats
            return "float"
        elif triton_type[0] == "i":
            # TODO handle different sized ints
            return "int"
        raise ValueError(f"Bad Triton Type: {triton_type}")

    def gen_c_triton_signature_declarations(self):
        '''
        Generate the C declarations for all of the triton kernels

        Ex: extern void [kernel_name](...);
        '''
        triton_src_files = os.popen('grep -rl "@triton.jit" dump | xargs grep -L "def call(args):"').read().split()
        declarations = []
        for file_name in triton_src_files:
            with open(file_name) as f:
                contents = f.readlines()
            signature = ""
            sig_types = {}
            for line in contents:
                if line.startswith("def"):
                    signature = line
                elif "triton_meta" in line:
                    start = line.find("'signature': ") + len("'signature': ")
                    end = line.find("}", start)
                    sig_types = eval(line[start:end + 1])
            open_paren = signature.find("(")
            close_paren = signature.find(")")
            kernel_name = signature[4:open_paren]
            arg_names = signature[open_paren + 1:close_paren].split(", ")
            # remove type annotations if present
            arg_names = [i.split(" : ")[0] for i in arg_names]
            triton_types = [sig_types[arg_name] for arg_name in arg_names]
            tmp = []
            for t in triton_types:
                if t != "constexpr":
                    tmp.append(t)
            triton_types = tmp
            c_types = [self.get_c_type(triton_type) for triton_type in triton_types]
            declarations.append(f"extern void {kernel_name}({', '.join(c_types)}, int, int, int, int, int, int);\n")
        declarations.append("extern void addmm(int, int, int, void*, void*, void*, void*, int, int, int, int, int, int, int, int, float, float, int, int, int, int, int, int);\n")
        return declarations

    def consume_buffer(self, string, start=0):
        '''Return the name of the next buffer in string the index of the end of the buffer name. Handles both plain buffer names and reinterpret_tensor calls'''

        string = string[start:]
        if string.startswith("reinterpret_tensor"):
            # reinterpret_tensor(bufx, (...shape), (...strides), y)
            func_name_len = len("reinterpret_tensor(")
            name = string[func_name_len:].split(", ")[0]
            # there are three close parens in the expression, find the last one
            idx = string.find(")")
            idx = string.find(")", idx + 1)
            idx = string.find(")", idx + 1)

            # i hate string parsing
            remaining_args = eval(string[func_name_len + len(name) + 2:idx])
            size = remaining_args[0]
            stride = remaining_args[1]
            offset = remaining_args[2]
            assert(offset == 0)
            return Buffer(name, shape=size, stride=stride, shadow=True), idx + start + 1
        else:
            # just the plain buffer name
            tokens = string.split(", ")
            if len(tokens) > 0 and ")" not in tokens[0]:
                # bufx, ...
                buf = tokens[0]
            else:
                # bufx)
                buf = string.split(")")[0]
            return self.buffers[buf], start + len(buf)

    def gen_c_triton_wrapper(self, python_wrapper):
        '''Convert the python wrapper that launches triton kernels into equivalent C code'''

        # map primals_n -> primals[n - 1]
        num_primals = len(self.collect_parameters(self.model)) + 1
        for i in range(num_primals):
            for j in range(len(python_wrapper)):
                python_wrapper[j] = python_wrapper[j].replace(f"primals_{i + 1}", f"primals[{i}]")

        c_wrapper = []
        c_wrapper.append("void call(float** primals, float* out) {\n")
        for line in python_wrapper:
            tokens = line.split()
            if len(tokens) >= 2 and tokens[-1] == "args" and tokens[-2] == "=":
                # primals_1, primals_2, ... = args
                for i in range(num_primals):
                    self.buffers[f"primals[{i}]"] = Buffer(f"primals[{i}]", shadow=True) # don't want to free
            elif tokens[0] == "args.clear()":
                continue
            elif tokens[0].startswith("assert_size_stride"):
                # assert_size_stride(primals_x, (size), (stride))
                name = tokens[0].split("(")[1][:-1]
                size, stride = eval(line[len(tokens[0] + " "):-2])
                self.buffers[name].shape = size
                self.buffers[name].stride = stride
            elif tokens[0] == "streamNone":
                continue
            elif len(tokens) >= 3 and tokens[2].startswith("reinterpret_tensor"):
                # bufx = reinterpret_tensor(bufy, ...); del bufx  # reuse
                self.buffers[tokens[0]], _ = self.consume_buffer(line, len(tokens[0] + " = "))
                c_wrapper.append(f"\tvoid* {tokens[0]} = {tokens[2][19:][:-1]};\n")
            elif len(tokens) >= 3 and tokens[2].startswith("empty_strided_cpu"):
                # bufx = empty_strided_cpu((a, b, c, ...), (stride), torch.float32)
                start_size_tuple = line.find("(") + 1
                end_size_tuple = line.find(")")
                end_stride_tuple = line.find(")", end_size_tuple + 1) + 1
                shape, stride = eval(line[start_size_tuple:end_stride_tuple])
                # shape = [int(i) if i != '' else 1 for i in line[start_size_tuple:end_size_tuple].split(", ")]
                size = 1
                for x in shape:
                    size *= x
                size *= self.get_torch_type_size(tokens[-1][:-1])
                c_wrapper.append(f"\tvoid* {tokens[0]} = heap_alloc({size});\n")
                self.buffers[tokens[0]] = Buffer(tokens[0], shape=shape, stride=stride, shadow=False)
            elif ".run" in tokens[0]:
                #[kernel_name].run(arg1, arg2, ..., stream=streamNone)
                kernel_name = tokens[0].split(".")[0]
                start_args = line.find("(") + 1
                end_args = line.find(")")
                args = line[start_args:end_args].split(", ")[:-1]
                # args = [arg if not arg.startswith("primals") else f"primals[{int(arg[8:]) - 1}]" for arg in args]
                c_wrapper.append(f"\t{kernel_name}({", ".join(args)}, 0, 0, 0, 1, 1, 1);\n")

            elif tokens[0].startswith("extern_kernels"):
                #extern_kernels.[kernel_name](...)
                c_wrapper.append(self.gen_extern_kernel_call(line))

            elif tokens[0].startswith("buf") and tokens[2].startswith("buf"):
                c_wrapper.append(f"\tvoid* {tokens[0]} = {tokens[2].replace(';', '')};\n")
            elif tokens[0] == "return":
                # return (reinterpret_tensor(bufx, ...), ...)
                # return (bufx, ...)
                buf, idx = self.consume_buffer(line[len("return ("):])
                c_wrapper.append(f"\tmemcpy(out, {buf.name}, {self.output_size_flat} * sizeof (float));\n")
                for name, buf in self.buffers.items():
                    if not buf.shadow:
                        c_wrapper.append(f"\theap_free({name});\n")
                c_wrapper.append("\treturn;\n")
        c_wrapper.append("}\n")
        return c_wrapper

    def gen_extern_kernel_call(self, line):
        '''Convert python code of the form extern_kernels.[kernel](...) to equivalent C call'''

        kernel_name = line.split(".")[1].split("(")[0]
        idx = line.find("(") + 1
        if kernel_name == "addmm":
            # extern_kernels.addmm(input, mat1, mat2, alpha=1.2, beta=0.7, out=out)
            input, idx = self.consume_buffer(line, idx)
            mat1, idx = self.consume_buffer(line, idx + len(", "))
            mat2, idx = self.consume_buffer(line, idx + len(", "))
            idx += len(", ")
            kwargs_str = line[idx:].split(", ")
            alpha = kwargs_str[0].split("=")[1]
            beta = kwargs_str[1].split("=")[1]
            out, _ = self.consume_buffer(kwargs_str[2].split("=")[1])
            for m in [mat1, mat2, input, out]:
                print(m.name, m.shape, m.stride)
            print()
            strides = ", ".join([f"{m.getStride(0)}, {m.getStride(1)}" for m in [mat1, mat2, input, out]])
            return f"pr_info(\"==== %d/100 %d/100\", (int)(((float*){out.name})[0]), (int)(((float*){out.name})[1]));\n\taddmm({out.getShape(0)}, {out.getShape(1)}, {mat1.getShape(1)}, {mat1.name}, {mat2.name}, {input.name}, {out.name}, {strides}, {alpha}, {beta}, 0, 0, 0, 1, 1, 1);\npr_info(\"!!!! %d/100 %d/100\", (int)(((float*){out.name})[0]), (int)(((float*){out.name})[1]));\n"
        else:
            raise Exception(f"Kernel {kernel_name} not implemented yet!")


    def copy_build_files(self, dir='build'):
        os.system(f'cp build_template/* {dir}')

    def build(self, output_file="build/main.c", debug=False):
        self.dump_torch_files()
        self.save_triton_kernels()
        self.copy_build_files()
        with open(output_file, 'w') as f:
            f.write("""#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/string.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>

MODULE_LICENSE("GPL");

#define heap_alloc(x) kmalloc(x, GFP_ATOMIC)
#define heap_free(x) kfree(x)

""")
            f.writelines(self.gen_c_triton_signature_declarations())
            f.writelines(self.gen_c_triton_wrapper(self.get_python_triton_wrapper()))
            f.write(self.gen_primals_init_code(torch.zeros(self.input_shape, dtype=torch.float32)))
            f.write(f"""

#define DEVICE_NAME "model_run"
#define CLASS_NAME  "model_class"

#define IOCTL_PROC _IOWR('m', 1, struct model_data *)

struct model_data {{
    float __user *in;   // pointer to user input buffer
    float __user *out;  // pointer to user output buffer
    int input_size;     // number of floats
}};

static dev_t dev_num;
static struct cdev my_cdev;
static struct class *my_class;

primals_t primals;

void forward(primals_t p, void* input, void* output) {{
    memcpy(p.primals[p.input_idx], input, p.input_size * sizeof(void*));
    call(p.primals, output);
}}

static long my_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{{
    struct model_data data;
    float kbuf_in[256];
    int i;

    if (cmd != IOCTL_PROC)
        return -EINVAL;

    if (copy_from_user(&data, (void __user *)arg, sizeof(data)))
        return -EFAULT;

    if (data.input_size > 256)
        return -EINVAL;

    // copy input floats from user
    if (copy_from_user(kbuf_in, data.in, sizeof(float) * data.input_size))
        return -EFAULT;

    float* kbuf_out = heap_alloc({self.output_size_flat} * sizeof (float));
    kernel_fpu_begin();
    forward(primals, kbuf_in, kbuf_out);
    kernel_fpu_end();

    // copy results back to user buffer
    if (copy_to_user(data.out, kbuf_out, {self.output_size_flat} * sizeof(float))) {{
        heap_free(kbuf_out);
        return -EFAULT;
    }}

    heap_free(kbuf_out);
    return 0;
}}

static struct file_operations fops = {{
    .owner          = THIS_MODULE,
    .unlocked_ioctl = my_ioctl,
}};



int init(void) {{
    pr_info("Hello %x!\\n", IOCTL_PROC);

    alloc_chrdev_region(&dev_num, 0, 1, DEVICE_NAME);
    cdev_init(&my_cdev, &fops);
    cdev_add(&my_cdev, dev_num, 1);
    my_class = class_create(CLASS_NAME);
    device_create(my_class, NULL, dev_num, NULL, DEVICE_NAME);

    primals = allocate_primals();
    /*
    kernel_fpu_begin();
    float* out = call(primals.primals);
    pr_info("%d/10000 %d/10000\\n", (int)(out[0] * 10000), (int)(out[1] * 10000));
    kernel_fpu_end();
    heap_free(out);
    */
    return 0;
}}

void cleanup(void) {{
    free_primals(primals);

    device_destroy(my_class, dev_num);
    class_destroy(my_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);

    pr_info("kernel module goodbye\\n");
}}
module_init(init);
module_exit(cleanup);
""")

        # os.system("rm -rf dump")
