from setuptools import setup
from setuptools import find_packages
from os.path import dirname, abspath, join
this_dir = abspath(dirname(__file__))

with open(join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gbdnn',
    version='0.1.0',
    description='https://github.com/huhuhhhh/gbdnn.git',
    long_description=long_description,
    long_description_content_type='text/markdown',
    download_url='https://github.com/huhuhhhh/gbdnn',
    license='BSD',
    install_requires=['numpy', 'scipy', 'matplotlib', 'monty', 'scikit-learn',
                      'tensorflow', 'keras'],
    packages=find_packages(),
    package_data={
        "gbdnn": ["*.json", "*.md"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 1 - alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mechanics",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)