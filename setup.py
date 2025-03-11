import os
import sys


try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst', format='md')
except (IOError, ImportError):
    long_description = open('README.md', encoding='utf-8').read()

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    sys.stderr.write('Setuptools not found!\n')
    raise

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except:
    sys.stderr.write('Pybind11 include not found')
    raise


extra_args = ['-std=c++14', '-march=native', '-O3']
extra_link_args = []

if sys.platform != 'darwin':
    extra_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']
else:
    extra_args += ['-mmacosx-version-min=10.9', '-stdlib=libc++', '-Xclang', '-fopenmp']
    extra_link_args += ['-lomp']
    os.environ['LDFLAGS'] = '-mmacosx-version-min=10.9'

module = Extension(
    'puffinn',
    sources=['python/wrapper/python_wrapper.cpp'],
    extra_compile_args=extra_args,
    extra_link_args=extra_link_args,
    include_dirs=['include', 'libs', pybind11_include])

setup(
    name='PUFFINN',
    version='0.2',
    author='Michael Erik Vesterli, Martin Aumüller, Matteo Ceccarello',
    author_email='maau@itu.dk',
    url='https://github.com/',
    description=
    'High-Dimenional Similarity search with guarantees based on Locality-Sensitive Hashing (LSH)',
    long_description=long_description,
    license='MIT',
    keywords=
    'nearest neighbor search similarity lsh locality-sensitive hashing cosine distance closest pair',
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[module])
