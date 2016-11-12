from setuptools import setup, find_packages
import os

setup(
    # Application name:
    name="tensorbayes",

    # Version number:
    version="0.01",

    # Application author details:
    author="Rui Shu",
    author_email="ruishu@stanford.edu",

    # Packages
    packages=find_packages(),

    # Data files
    include_package_data=True,
    zip_safe=False,

    # Details
    url="http://www.github.com/RuiShu/kaos",

    license="LICENSE.md",
    description="a library designed to simplify the building of deep amortized inference models",
    keywords='tensorflow deep learning',

    install_requires = ['numpy','keras','tensorflow'],

    entry_points = {
        'console_scripts': [
            'tensorbayes=tensorbayes.scripts:main',
        ],
    },

)
