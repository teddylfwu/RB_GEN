from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
	Extension("rb_gen",
		["rb_gen.pyx", "feature_matrix.cxx", "../generate_grids.c", "../random.c"],
		include_dirs=["..", numpy.get_include()]
	)
]

setup(name='Random Binding',
      ext_modules=cythonize(extensions))
