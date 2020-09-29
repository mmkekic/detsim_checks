import numpy
from Cython.Compiler.Options import get_directive_defaults
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
directive_defaults = get_directive_defaults()

# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True
from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(Extension("*", sources=["*.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=["-O3"], define_macros=[('CYTHON_TRACE', '1')])))
