VERSION = "0.6.1"

from setuptools import setup, find_packages
import sys

print(f"VERSION : {VERSION}")
DESCRIPTION = 'Low altitude aircraft anomaly detector'
LONG_DESCRIPTION = """
    This package contains a trained model to detect anomalies in low altitude traffic.
    The model focus on spoofing attack, by recognizing the aircraft the type based on its ADS-B signal.
    Hence, if the aircraft pretend to be a small aircraft, but the model recognize it as a plane, it is an anomaly.
"""

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="AdsbAnomalyDetector",
    version=VERSION,
    author="Pirolley Melvyn",
    author_email="",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
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