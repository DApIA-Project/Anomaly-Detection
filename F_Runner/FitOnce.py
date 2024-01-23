# Mlflow log-in
import _Utils.mlflow as mlflow

# Convert CTX to dict for logging hyperparameters
from _Utils.module import module_to_dict 
import numpy as np

# For auto-completion, we use Abstract class as virtual type
from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as _Trainer_


def fitOnce(Model:"type[_Model_]", Trainer:"type[_Trainer_]", CTX, default_CTX=None, experiment_name:str = None):
    """
    Fit the model once with a given set of hyperparameters.

    Parameters:
    -----------
    model: type[Model]:
        Model used for training

    trainer: type[Trainer]
        Trainer class, managing the training loop, testing and evaluation, for a specific task 
        (eg. spoofing detection)

    CTX: Module 
        Python module containing the set of hyperparameters
    """

    # Init mlflow (can be ignored if mlflow is not used)
    run_number = mlflow.init_ml_flow(experiment_name)
    run_name = str(run_number) + " - " + Model.name
    print("Run name : ", run_name)
    
    # Convert CTX to dict and merge it with default_CTX
    CTX = module_to_dict(CTX)
    if (default_CTX != None):
        default_CTX = module_to_dict(default_CTX)
        for param in default_CTX:
            if (param not in CTX):
                CTX[param] = default_CTX[param]

    print(CTX["USED_FEATURES"])

    with mlflow.start_run(run_name=run_name) as run:
        for param in CTX:
            if (type(CTX[param]) == bool): # Convert bool to int the make boolean hyperparameters visualisable in mlflow
                mlflow.log_param(param, int(CTX[param]))
            else:
                mlflow.log_param(param, CTX[param])

        # Create a new training environment and run it
        trainer = Trainer(CTX, Model)
        metrics = trainer.run()

        # Log the result metrics to mlflow
        for metric_label in metrics:
            value = metrics[metric_label]
            mlflow.log_metric(metric_label, value)
