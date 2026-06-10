
import pathlib
import tempfile

from .gen_kernel_module import TorchKernelDeployer


def model_compile(model, input_shape):
    tkd = TorchKernelDeployer(model, input_shape)
    files = []
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="torch_kernmod_"))
    build_path = pathlib.Path(tmp)
    tkd.build(build_path)
    for build_file in build_path.glob("*"):
        files.append(build_file)
    return files
