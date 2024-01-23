# ADSB Anormaly Detector Manual

## Installation

### Recommended
The library can be installed by using python's package manager pip (recommended):

```
pip install AdsbAnomalyDetector
```
### Re-build the lib
You can also choose to build the library by yourself. It can be interesting in case you want to train your own AI model.

After training the model, run the build.py file to build the library.

## Tutorial

### Check spoofing anomalies

In order to check if an aircraft isn't spoofing it's identity, the first step is to retrieve the aircraft type based on its trajectory and then to check that it correspond to the pretended identity of the aircraft.

To achieve that, the library provide the **predictAircraftType** function that run an artificial intelligence model trained to retrieve the model's type based on it's trajectory.

| function | parameters | returns |
|---|---|---|
| predictAircraftType(messages) -> dict | __message__ : list of messages. | a |

