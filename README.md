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

## Experiment 4: TrajectorySeparator

Separate messages having duplicated identification code (when the system is under flooding).
Allow to reconstruct the trajectories of ghost aircraft to find which message are fake with our FloodingSolver module.

## lib compilation

```cd _lib```

```python build.py sdist bdist_wheel```

```pip install dist/AircraftClassifier-0.0.1-py3-none-any.whl```

