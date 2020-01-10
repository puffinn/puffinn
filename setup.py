import os
import sys


try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst', format='md')
except (IOError, ImportError):
    long_description = open('README.md').read()

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    sys.stderr.write('Setuptools not found!\n')
    raise

# native clang doesn't support openmp
# TODO add better way to check for openmp
use_openmp = sys.platform != 'darwin'
extra_args = ['-std=c++14', '-march=native', '-O3']
extra_link_args = []

if use_openmp:
    extra_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']
if sys.platform == 'darwin':
    extra_args += ['-mmacosx-version-min=10.9', '-stdlib=libc++']
    os.environ['LDFLAGS'] = '-mmacosx-version-min=10.9'

module = Extension(
    'puffinn',
    sources=['python/wrapper/python_wrapper.cpp'],
    extra_compile_args=extra_args,
    extra_link_args=extra_link_args,
    include_dirs=['include', 'external/pybind11/include', 'libs'])

setup(
    name='PUFFINN',
    version='0.1',
    author='Michael Erik Vesterli, Martin Aumüller',
    author_email='maau@itu.dk',
    url='https://github.com/',
    description=
    'High-Dimenional Similarity search with guarantees based on Locality-Sensitive Hashing (LSH)',
    long_description=long_description,
    license='MIT',
    keywords=
    'nearest neighbor search similarity lsh locality-sensitive hashing cosine distance',
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[module])
