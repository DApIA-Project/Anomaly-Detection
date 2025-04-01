
from _Utils.DebugGui import launch_gui

# Convert CTX to dict for logging hyperparameters
from _Utils.module import buildCTX
from numpy_typing import np, ax

# For auto-completion, we use Abstract class as virtual type
from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as _Trainer_

from _Utils.RunLogger import RunLogger

RUN_LOGGER = RunLogger("./_Artifacts/logs.pkl")


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

    # Convert CTX to dict and merge it with default_CTX
    CTX = buildCTX(CTX, default_CTX)

    # Create a new training environment and run it
    launch_gui(CTX)
    trainer = Trainer(CTX, Model)
    
    metrics = trainer.run()
    groups = {}
    dtypes = {}
    
    print(metrics)
    
    
    for key in metrics.keys():
        groups[key] = "Metric"
        
    exept = ["MAX_BATCH_SIZE", "MIN_BATCH_SIZE", "TEST_RATIO"]
    for k in CTX:
        if (k in exept):
            continue
        # if the variable is numeric
        if (isinstance(CTX[k], (int, float, bool))):
            groups[k] = "HYPERPARAMETERS"
            metrics[k] = CTX[k]
        if (isinstance(CTX[k], str)):
            groups[k] = "STR_PARAMETERS"
            metrics[k] = CTX[k]
        
    metrics["model"] = Model.name
    
    # get path
    metrics["PROBLEM"] = experiment_name
    groups["PROBLEM"] = "PROBLEM"
    dtypes["PROBLEM"] = str
    dtypes["STR_PARAMETERS"] = str
    
    # if (metrics["EPOCHS"] != 0):
    RUN_LOGGER.add_run(metrics, groups, dtypes)
    
    loggers_per_problem = RUN_LOGGER.split_by("PROBLEM")
    
    for i in range(len(loggers_per_problem)):
        # if (loggers_per_problem[i].get("PROBLEM") == "AircraftClassification"):
        loggers_per_problem[i] = loggers_per_problem[i].get_best_groupes_by("ACCURACY", "model", maximize=True)
    loggers_per_problem:RunLogger = RunLogger.join(loggers_per_problem)
    
    file = open("./_Artifacts/logs.txt", "w")
    loggers_per_problem.group_by("PROBLEM").render(file, "Best models")
    RUN_LOGGER.group_by("PROBLEM").render(file, "All runs")
    file.close()
    
    
    
    


