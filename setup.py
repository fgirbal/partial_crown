import sys
from os import path
from setuptools import setup, find_packages

repository_dir = path.dirname(__file__)

with open(path.join(repository_dir, "requirements.txt")) as fh:
    requirements = [line for line in fh.readlines()]

setup(
    name='PINN-Verifier',
    version='0.0.1',
    description='Physics-informed neural networks certification framework',
    author='Francisco Girbal Eiras',
    license='MIT',
    python_requires=">=3.8",
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
)
