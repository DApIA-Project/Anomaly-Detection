# Anomaly-Detection
Deep learning algoritms to detect anomaly inside ADS-B data.


## Experiment 1: AicraftClassification

ADS-B anomaly detector based on the fact that aircraft trajectories "shapes" should match the aircraft's type given
by it's ICAO code.

Along the type of the aircrat, trajectories contains reccurent patterns wich can be identify by a AI to
determine the aircraft type.

The goal of this approach is to prevent Spoofing attack by detecting if the icao/callsing of an aircraft is correct, by checking they are correcponding to his trajectory "shape".


## Experiment 2: ReplaySolver

Allow the detection of some Ghost aircraft by checking if a trajectory has already happened in the past.
The approach uses fingerprint and hashshing and aim to find if a given slice of a trajectory
is already in our hash table or not.
If it is, it mean the trajectory is a replay.

This approach has the benefit of being invarient to geometric transformations (scaling, rotation, translation ...)
And it is very fast, even with very large database thanks to the usage of an hash table.

## Experiment 3: FloodingSolver

Detect flooding attack by checking the coherence of ghost aircrafts trajectories.
The approach uses a deep learning model to predict the next position of an aircraft given his historical trajectory.
If the predicted position is too far from the real position, it means the model is "surprised" by the trajectory and it is likely to be a fake one.

## Experiment 4: TrajectorySeparator

Separate messages having duplicated identification code (when the system is under flooding).
Allow to reconstruct the trajectories of ghost aircraft to find which message are fake with our FloodingSolver module.

## How to make detection with you own files ?

Install our compiled library with pip

```pip install AdsbAnomalyDetector```


Then you will be able to run the codes in the ```_Examples``` folder.
So go to the ```_Examples``` folder and run our webserver example with :

```python webserver.py```

Then startup our visualizer available at [https://adsb-visualizer.web.app/](https://adsb-visualizer.web.app/)
Drop any ADS-B file you want to check in the visualizer. It will automatically send the right request to your webserver and display the results with green trajectories for normal aircrafts and red ones for anomalies.


Take care of not running multiple instances of the visualizer at the same time, as it will cause the webserver to crash or behave unexpectedly.



## Re-build the library

Novigate to the ```_lib``` folder and build script to create the library from the source files.

```python build.py```

Then you can install the library with pip

```pip install dist/AdsbAnomalyDetector-{a.b.c}-py3-none-any.whl```

