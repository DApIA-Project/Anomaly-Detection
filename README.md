# Instructions

## Installation

Clone the repository :

```
git clone https://github.com/DApIA-Project/Anomaly-Detection/
cd ./Anomaly-Detection
git checkout OpenSky
```

Install the dependencies:

- tensorflow
- pandas
- matplotlib
- pydot
- scikit-learn
- pickle-mixin

## Dataset

The dataset is available in the following link: 
https://mega.nz/folder/R1MHER6a#uRYHrlwbAb14JCqoHDlfwg

Drop the folder ```AircraftClassification/``` directly into the ```A_Dataset/``` folder


## Execute the code

To run the code, you can simply use the main.py file:
```python main.py```

For faster execution, we recommend the usage of GPU.

## Get the results

After the execution, several analytics files will be generated.
In ```_Artefact/``` you will find loss and accuracy curves, evaluation and per-timestep confusion matrix, and finally models mistakes in ```_Artefact/eval.pdf```

The model's predictions are detailed in ```A_Dataset/AircraftClassification/Outputs/Eval/```.
You can visualize those dragging the .csv files into our dedicated ADS-B visualizer: https://adsb-visualizer.web.app/
Safe trajectories are displayed in Green whereas suspicious trajectories are displayed in Red.  
