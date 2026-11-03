# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT_DIR = Path(__file__).resolve().parent
EXT_SRC_ROOT = ROOT_DIR / "_ext_src"
SRC_DIR = EXT_SRC_ROOT / "src"
INCLUDE_DIR = EXT_SRC_ROOT / "include"

cpp_sources = sorted(SRC_DIR.glob("*.cpp"))
cuda_sources = sorted(SRC_DIR.glob("*.cu"))
ext_sources = cpp_sources + cuda_sources

required_headers = [
    "ball_query.h",
    "group_points.h",
    "interpolate.h",
    "sampling.h",
    "cuda_utils.h",
]

missing_headers = [
    str(INCLUDE_DIR / name)
    for name in required_headers
    if not (INCLUDE_DIR / name).is_file()
]

if missing_headers:
    raise FileNotFoundError(
        "Missing PointNet2 header files:\n"
        + "\n".join(missing_headers)
    )

if not ext_sources:
    raise FileNotFoundError(
        f"No C++ or CUDA sources found in: {SRC_DIR}"
    )

print("PointNet2 source directory:", SRC_DIR)
print("PointNet2 include directory:", INCLUDE_DIR)
print("Number of source files:", len(ext_sources))

setup(
    name="pointnet2",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="pointnet2._ext",
            sources=[str(path) for path in ext_sources],
            include_dirs=[str(INCLUDE_DIR)],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    "-O3",
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True)
    },
)