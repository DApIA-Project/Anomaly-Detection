# Anomaly-Detection
Deep learning algoritms to detect anomaly inside ADS-B data.

## Models and Experiments

### Experiment 1: AicraftClassification

ADS-B anomaly detector based on the fact that aircraft trajectories "shapes" should match the aircraft's type given
by it's ICAO code.

Along the type of the aircrat, trajectories contains reccurent patterns wich can be identify by a AI to
determine the aircraft type.

The goal of this approach is to prevent Spoofing attack by detecting if the icao/callsing of an aircraft is correct, by checking they are correcponding to his trajectory "shape".


### Experiment 2: ReplaySolver

Allow the detection of some Ghost aircraft by checking if a trajectory has already happened in the past.
The approach uses fingerprint and hashshing and aim to find if a given slice of a trajectory
is already in our hash table or not.
If it is, it mean the trajectory is a replay.

This approach has the benefit of being invarient to geometric transformations (scaling, rotation, translation ...)
And it is very fast, even with very large database thanks to the usage of an hash table.

### Experiment 3: FloodingSolver

Detect flooding attack by checking the coherence of ghost aircrafts trajectories.
The approach uses a deep learning model to predict the next position of an aircraft given his historical trajectory.
If the predicted position is too far from the real position, it means the model is "surprised" by the trajectory and it is likely to be a fake one.

### Experiment 4: TrajectorySeparator (Paused)

Separate messages having duplicated identification code (when the system is under flooding).
Allow to reconstruct the trajectories of ghost aircraft to find which message are fake with our FloodingSolver module.

### Experiment 5: TrajectoryChecker (Not started)

Check the coherence of a trajectory by using an autoencoder.
An abnormal trajectory would have a high reconstruction error.

# Download the dataset

The dataset is available at [https://zenodo.org/doi/10.5281/zenodo.10050766](https://zenodo.org/doi/10.5281/zenodo.10050766)


# Run the code

## How to make detection with you own files ?

The simplest way to use the models and start making prediction on your own files is to install our compiled library with pip

```bash
pip install AdsbAnomalyDetector
```

You can then import the library and start using the models.

Enter ```_Examples``` folder and run :

```bash
python webserver.py
```

It will start a webserver on your local machine, ready to receive ads-b messages and process them for anomaly detection.

You can then startup our visualizer available at [https://adsb-visualizer.web.app/](https://adsb-visualizer.web.app/)
Drop any ADS-B file you want to check in the visualizer. It will automatically send the right request to your webserver and display the results with green trajectories for normal aircrafts and red ones for anomalies.


Take care of not running multiple instances of the visualizer at the same time, as it will cause the webserver to crash or behave unexpectedly.

## How to use the AdsbAnomalyDetector library ?

You can find the library on [https://pypi.org/project/AdsbAnomalyDetector/](https://pypi.org/project/AdsbAnomalyDetector/)
Some informations will soon be available on this page.

You can use the library by importing it and calling the predict function with a list of ADS-B messages :

```python
from AdsbAnomalyDetector import predict, AnomalyType

messages = [ # list of flights messages at t = 1609459200
    {
        "icao": "3C4A4D",
        "callsign": "AFR123",
        "timestamp": 1609459200,
        "latitude": 48.8583,
        "longitude": 2.2945,
        "altitude": 10000,
        "geoaltitude": 10000,
        "velocity": 250,
        "vertical_rate": 0,
        "track": 90,
        "alert": False,
        "spi": False,
    },
    {
        "icao": "39AC45",
        "callsign": "SAMU31",
        "timestamp": 1609459200,
        ...
    }, ...
]

messages = predict(messages)
```

The predict function will return the same list of messages with an additional field "anomaly" that will be ```AnomalyType.SPOOFING | AnomalyType.REPLAY | AnomalyType.FLOODING | AnomalyType.VALID``` depending on the anomaly detected.

The predict function should be called once every timestamp and should not be called on various timestamps at the same time.

## Re-build the library

Navigate to the ```_lib``` folder and build script to create the library from the source files.

```bash
python build.py
```

Then you can install the library with pip

```bash
pip install dist/AdsbAnomalyDetector-{a.b.c}-py3-none-any.whl
```

## Code Tree

```bash
├── _Artifacts # debug files generation (charts, logs, ...)
│   └── ${Experiment Name}
│       └──${Model Name}
│           └──... # whatever you want to save
├── _Examples
│   ├── webserver.py # example of webserver to run the anomaly detection
│   └── ... # other examples
├── _Lib
│   ├── AdsbAnomalyDetector # contains needed files library (auto-generated by build.py)
│   ├── dist #  wheel of the library (auto-generated by build.py)
│   ├── setup.py # setup file to make the wheel (called by build.py)
│   └── build.py # script that regroup needed files for the library
├── A_Dataset # dataset used for training and testing
│   └── ${Experiment Name}
│       └──${Model Name}
│           ├── Train
│           └── Eval
├── B_Models
│   ├── ${Experiment Name}
│   │   └──${Model Name}.py
│   └── Utils # utility functions for the models (custom layers, metrics, ...)
│       └──...
├── C_Constants
│   └── ${Experiment Name}
│       └──${Model Name}.py
├── D_DataLoader
│   ├── ${Experiment Name}
│   │   ├── DataLoader.py
│   │   └── Utils.py # specific utility functions for the dataloader
│   └── Utils.py # global utility functions for dataloaders
├── E_Trainer
│   ├── ${Experiment Name}
│   │   └──Trainer.py
│   └── AbstractTrainer.py # abstract class for the trainers
├── F_Runner
│   ├── FitOnce.py # run a single training
│   └── MultiFit.py # run multiple training to check stability
├── G_Main
│   └── ${Experiment Name}
│       │ # main file to run the experiment with a specific model :
│       └──exp_${Model Name}.py
├── main.py # pick an experiment to run and start it
├── .gitignore
└── README.md
```