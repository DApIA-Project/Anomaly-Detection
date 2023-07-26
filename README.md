# Anomaly-Detection
Deep learning approach to detect anomaly inside ADS-B data.


## Experiment 1: AicraftClassification (In progress)

Aircraft classification based on type (Commercial planes, Tourism plane, Helicopter ...).
The goal of this approach is to prevent "spoofing" attack by to detecting if the icao/callsing of an aircraft is correcponding to his trajectory.



## lib compilation

```cd _lib```

```python build.py sdist bdist_wheel```

```pip install dist/Anomaly_Detection-0.0.1-py3-none-any.whl```