VERSION = "0.6.1"

from setuptools import setup, find_packages
import sys

print(f"VERSION : {VERSION}")
DESCRIPTION = 'Low altitude aircraft anomaly detector'

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="AdsbAnomalyDetector",
    version=VERSION,
    author="Pirolley Melvyn",
    author_email="",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=["tensorflow", "numpy", "pandas", "scikit-learn", "matplotlib", "pickle-mixin"], # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    package_data={
        '': ['*.py', "map.png", "labels.csv", "./AircraftClassification/*", "./FloodingSolver/*"],
    },
    keywords=['python', 'deep learning', 'tensorflow', 'aircraft', 'classification', 'ADS-B'],
    classifiers= [
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)