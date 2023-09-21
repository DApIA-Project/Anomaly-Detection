# Anomaly-Detection
Deep learning approach to detect anomaly inside ADS-B data.


## Experiment 1: AicraftClassification (In progress)

Aircraft classification based on type (Commercial planes, Tourism plane, Helicopter ...).
The goal of this approach is to prevent "spoofing" attack by to detecting if the icao/callsing of an aircraft is correcponding to his trajectory.



## lib compilation

```cd _lib```

```python build.py sdist bdist_wheel```

```pip install dist/AircraftClassifier-0.0.1-py3-none-any.whl```


## results

<!-- make a table -->
| Model | Accuracy |
| --- | --- |
| absolute pos, no map | 94.2 (mean) |
| Relative pos, no map | 95.6 (max) |
| absolute pos, + map | --- |
| Relative pos, + map | --- |

| Model | Accuracy |
| --- | --- |
| Without all ... | --- |
| with spi squawk alert... | --- |
| With timestamps | --- |
| With relative tracks | --- |


| Model | Accuracy |
| --- | --- |
| load last model | 93.7 (mean) 94.7 (max) |
| load "best" model | 95.6 |


| Model | Accuracy |
| --- | --- |
| 1 layers | --- |
| 2 layers | --- |
| 3 layers | --- |

| Model | Accuracy |
| --- | --- |
| 32 batch S | --- |
| 64 batch S  | --- |
| 128 batch S  | --- |


| Model | Accuracy |
| --- | --- |
| training noise | --- |
| training noise | --- |
| training noise | --- |

| Model | Accuracy |
| --- | --- | 
| NB_TRAIN_SAMPLES 10 | --- |
| NB_TRAIN_SAMPLES 5 | --- |
| NB_TRAIN_SAMPLES 1 | --- |


| Model | Accuracy |
| --- | --- | 
| learning rate 10-3 | 95.6 |
| learning rate 8e10-4 | --- |
| learning rate 6e10-4 | 93.2 |



| Model | Accuracy |
| --- | --- | 
| separated takeoff and adsb training | --- |
| all together | --- |



| Model | Accuracy |
| --- | --- | 
| use repeat vector injection at the middle of the CNN to merge ctx | --- |
| merge at the end in dense | --- |