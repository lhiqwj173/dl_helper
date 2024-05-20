import os
import pybind11
import numpy as np
from setuptools import setup, Extension

cpp_ext_module = Extension(
    'cpp_ext',
    sources=['cpp_ext.cpp'],
    include_dirs=[pybind11.get_include(), np.get_include()],
    language='c++',
    extra_compile_args=['/std:c++17']
)

setup(
    ext_modules=[cpp_ext_module],
)
