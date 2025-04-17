import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import textwrap


class ModelConverter:
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
        self.model = model
        self.input_shape = input_shape
        self.weights_data = []
        self.layer_dims = []
        self.activation_types = []

    def analyze_model(self):
        """Extract model architecture and weights"""
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                weights = layer.weight.data.numpy()
                bias = layer.bias.data.numpy()
                self.weights_data.extend([weights, bias])
                self.layer_dims.append((weights.shape[1], weights.shape[0]))

            elif isinstance(layer, nn.ReLU):
                self.activation_types.append("relu")
            # Add more layer types as needed

    def generate_weight_arrays(self) -> str:
        """Generate C code for weight arrays"""
        code = []
        total_weights = 0

        for i, weights in enumerate(self.weights_data):
            flat_weights = weights.flatten()
            total_weights += len(flat_weights)

            # Format weights array
            values = [f"{w:.6f}" for w in flat_weights]
            array_str = textwrap.fill(", ".join(values), width=80)

            if i % 2 == 0:  # Weight matrix
                code.append(
                    f"static const float weights{i//2}[] = {{\n{array_str}\n}};"
                )
            else:  # Bias vector
                code.append(f"static const float bias{i//2}[] = {{\n{array_str}\n}};")

        return "\n\n".join(code)

    def generate_forward_pass(self) -> str:
        """Generate C code for forward pass computation"""
        code = []
        code.append(
            """
static void forward(const float* input, float* output) {
    float* temp;
    float* temp1 = (float*)kmalloc(sizeof(float) * MAX_LAYER_SIZE, GFP_KERNEL);
    float* temp2 = (float*)kmalloc(sizeof(float) * MAX_LAYER_SIZE, GFP_KERNEL);
    float *curr = temp1, *next = temp2;"""
        )

        # First layer uses input directly
        dims = self.layer_dims[0]
        code.append(
            f"""
    // Layer 0
    for(int i = 0; i < {dims[1]}; i++) {{
        float sum = 0.0f;
        for(int j = 0; j < {dims[0]}; j++) {{
            sum += input[j] * weights0[i * {dims[0]} + j];
        }}
        curr[i] = sum + bias0[i];"""
        )

        if self.activation_types and self.activation_types[0] == "relu":
            code.append("        curr[i] = curr[i] > 0.0f ? curr[i] : 0.0f;")
        code.append("    }")

        # Middle layers
        for layer_idx in range(1, len(self.layer_dims) - 1):
            dims = self.layer_dims[layer_idx]
            code.append(
                f"""
    // Layer {layer_idx}
    for(int i = 0; i < {dims[1]}; i++) {{
        float sum = 0.0f;
        for(int j = 0; j < {dims[0]}; j++) {{
            sum += curr[j] * weights{layer_idx}[i * {dims[0]} + j];
        }}
        next[i] = sum + bias{layer_idx}[i];"""
            )

            if (
                layer_idx < len(self.activation_types)
                and self.activation_types[layer_idx] == "relu"
            ):
                code.append("        next[i] = next[i] > 0.0f ? next[i] : 0.0f;")
            code.append("    }")
            code.append("    temp = curr; curr = next; next = temp;")

        # Final layer
        last_idx = len(self.layer_dims) - 1
        if last_idx > 0:
            dims = self.layer_dims[last_idx]
            code.append(
                f"""
    // Output Layer
    for(int i = 0; i < {dims[1]}; i++) {{
        float sum = 0.0f;
        for(int j = 0; j < {dims[0]}; j++) {{
            sum += curr[j] * weights{last_idx}[i * {dims[0]} + j];
        }}
        output[i] = sum + bias{last_idx}[i];
    }}"""
            )

        code.append(
            """
}"""
        )
        return "\n".join(code)

    def generate_c_code(self) -> str:
        """Generate complete C code"""
        # Find maximum layer size for temp buffer allocation
        max_layer_size = max(dim[1] for dim in self.layer_dims)

        header = f"""#include <linux/module.h>
#include <linux/module.h> /* Needed by all modules */ 
#include <linux/printk.h> /* Needed for pr_info() */ 
#include <linux/slab.h>
#include <asm/fpu/api.h>
#include <linux/timex.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#define MAX_LAYER_SIZE {max_layer_size}
#define INPUT_SIZE {self.input_shape[0]}
#define OUTPUT_SIZE {self.layer_dims[-1][1]}
"""

        # Generate complete code
        return "\n".join(
            [
                header,
                self.generate_weight_arrays(),
                self.generate_forward_pass(),
                """

// output print function
void printArr(float* arr, int n) {
    for (int i = 0; i < n; i++){ 
          float x = arr[i];
          int int_x = (int) x;
          float dec_x_temp = (x-int_x)*1000;
          int dec_x = (int) dec_x_temp;
        pr_info("%u.%u", int_x, dec_x);
        pr_info("\\n");
    }
}

// Example usage function
int __init init_module(void){
    pr_info("Kernel Module Init \\n");
    kernel_fpu_begin();
    float input [INPUT_SIZE];
    float output [OUTPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++){
	    input [i] = 0.1;
    }

    struct timespec64 t0,t1;
    ktime_get_real_ts64(&t0);

	forward(input, output);

    ktime_get_real_ts64(&t1);

    long long ns = (t1.tv_sec - t0.tv_sec) * 1000000000ULL + (t1.tv_nsec - t0.tv_nsec);
    pr_info("%lld ns", ns);

    //printArr(output, OUTPUT_SIZE);
    kernel_fpu_end();
    return 0;
}

void __exit cleanup_module(void)
{

	pr_info("Goodbye \\n");
}

MODULE_LICENSE("GPL");
                """,
            ]
        )


# Example usage
def convert_model(model_path: str, input_shape: Tuple[int, ...], output_file: str):
    # Load model
    model = torch.load(model_path, weights_only=False)
    model.eval()

    # Convert model
    converter = ModelConverter(model, input_shape)
    converter.analyze_model()
    c_code = converter.generate_c_code()

    # Save C code
    with open(output_file, "w") as f:
        f.write(c_code)

    return c_code


# Example test code
if __name__ == "__main__":
    # Create a simple test model
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(12, 256)
            self.relu = nn.ReLU()

            self.fc2 = nn.Linear(256, 256)
            self.relu1 = nn.ReLU()
            self.fc3 = nn.Linear(256, 256)
            self.relu2 = nn.ReLU()
            self.fc4 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu1(self.fc2(x))
        x = self.relu2(self.fc3(x))
        x = self.fc4(x)

        return x

    # Initialize and save test model
    # model = SimpleNet()
    # torch.save(model, "test_model.pth")

    # Convert to C
    c_code = convert_model(
        "./test_model.pth",
        (12,),
        "model_kernel_mod_e2egen.c",
    )
    print("Generated C code saved to model_kernel_mod.c")
