import os
import pybind11
import numpy as np
from setuptools import setup, Extension
import platform

extra_compile_args = []
if platform.system() == 'Windows':
    extra_compile_args.append('/std:c++17')
else:
    extra_compile_args.append('-std=c++17')

cpp_ext_module = Extension(
    'cpp_ext',
    sources=['cpp_ext.cpp'],
    include_dirs=[pybind11.get_include(), np.get_include()],
    language='c++',
    extra_compile_args=extra_compile_args
)

setup(
    ext_modules=[cpp_ext_module],
)
