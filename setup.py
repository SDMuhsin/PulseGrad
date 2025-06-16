from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='experimental_ext',
    ext_modules=[
        CUDAExtension(
            'experimental_ext', [
                # We now only need to compile the binding file,
                # as it includes the other .cu file.
                'src/experimental/experimental_cuda_binding.cu',
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
