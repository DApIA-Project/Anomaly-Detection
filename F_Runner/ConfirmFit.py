


# import MLviz
# Mlflow logging
import _Utils.mlflow as mlflow

# Convert CTX to dict for logging hyperparameters
from _Utils.module import module_to_dict 
import numpy as np

# For auto-completion, we use Abstract class as virtual type
from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as _Trainer_



def confirm_fit(Model:"type[_Model_]", Trainer:"type[_Trainer_]", CTX, default_CTX=None, experiment_name:str = None):
    """
    Fit the model one time with the given CTX of hyperparameters

    Parameters:
    -----------
    model: type[Model]:
        Model class type of the model to train

    trainer: type[Trainer]
        Trainer class type of the trainer algorithm to use

    CTX: Module 
        Python module containing the hyperparameters and constants

    experiment_name: str 
        Name of the experiment to log to mlflow
    """


    
    # Init mlflow
    run_number = mlflow.init_ml_flow(experiment_name)
    run_name = str(run_number) + " - " + Model.name
    print("Run name : ", run_name)
    
    # MLviz.setExperiment(experiment_name)
    # MLviz.startRun(run_name)
    

    # Convert CTX to dict and log it
    CTX = module_to_dict(CTX)
    if (default_CTX != None):
        default_CTX = module_to_dict(default_CTX)
        for param in default_CTX:
            if (param not in CTX):
                CTX[param] = default_CTX[param]

    metrics_stats:dict[str,list] = {}

    print(default_CTX)

    for i in range(CTX["NB_TRAIN"]):
        with mlflow.start_run(run_name=run_name) as run:
            for param in CTX:
                if (type(CTX[param]) == bool): # Convert bool to int the make boolean hyperparameters visualisable in mlflow
                    mlflow.log_param(param, int(CTX[param]))
                else:
                    mlflow.log_param(param, CTX[param])

            
            # Instanciate the trainer and go !
            trainer = Trainer(CTX, Model)
            metrics = trainer.run()

            # Log the result metrics to mlflow
            for metric_label in metrics:
                value = metrics[metric_label]
                if (metric_label not in metrics_stats):
                    metrics_stats[metric_label] = [value]
                else:
                    metrics_stats[metric_label].append(value)
                mlflow.log_metric(metric_label, value)

        if (i == CTX["NB_TRAIN"] -1):
            print("Metric".rjust(15), "|", "min".rjust(10), "|", "mean".rjust(10), "|", "std".rjust(10), "|", "median".rjust(10), "|", "max".rjust(10), sep="")
            for metric in metrics:
                # min, mean, std, median, max
                _min = np.min(metrics_stats[metric])
                _mean = np.mean(metrics_stats[metric])
                _std = np.std(metrics_stats[metric])
                _median = np.median(metrics_stats[metric])
                _max = np.max(metrics_stats[metric])

                mlflow.log_metric(metric + "_min", _min)
                mlflow.log_metric(metric + "_mean", _mean)
                mlflow.log_metric(metric + "_std", _std)
                mlflow.log_metric(metric + "_median", _median)
                mlflow.log_metric(metric + "_max", _max)

                print(metric.rjust(15), "|", str(round(_min, 3)).rjust(10), "|", str(round(_mean, 3)).rjust(10), "|", str(round(_std, 3)).rjust(10), "|", str(round(_median, 3)).rjust(10), "|", str(round(_max, 3)).rjust(10), sep="")


