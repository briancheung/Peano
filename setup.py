import os
from setuptools import find_packages
from setuptools import setup

version = '0.01dev'

install_requires = [
                    'numpy',
                    'theano',
                    ]

setup(
    name='Peano',
    version=version,
    description='A library providing theano ops which also have learnable parameters.',
    author='Brian Cheung',
    author_email='bcheung@berkeley.edu',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    )