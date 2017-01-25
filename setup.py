# Always prefer setuptools over distutils
from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nanograin',
    version='0.0.1',
    description='Calculate energies of nano-sized alloys',
    long_description=long_description,
    url='https://gitlab.com/bocklund/nanograin',
    author='Brandon Bocklund',
    author_email='bocklund@psu.edu',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords='engineering materials sciece metallurgy nanomaterials',
    packages=['nanograin'],
    install_requires=['numpy', 'scipy', 'matplotlib'],
    extras_require={'test': ['pytest'],},
    package_data={'nanograin': ['materials-data/*.json']},
)
