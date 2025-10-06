from _Utils.RunLogger import RunLogger
from B_Model.AbstractModel import Model as _Model_


try:
    from _Utils import secrets_stuffs as S
    RUN_LOGGER = RunLogger(host=S.IP, port=S.PORT)
except ImportError:
    RUN_LOGGER = RunLogger("./_Artifacts/logs.pkl")
    print("Running in local mode, using local logger.")



def log_data(metrics, CTX, Model:"type[_Model_]", experiment_name:str = None):
    groups = {}
    dtypes = {}
    
<<<<<<< HEAD
    print(metrics)
=======
>>>>>>> master
    
    
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
    
    loggers_for_flooding_per_horizon = RUN_LOGGER.split_by("PROBLEM")
    flood_i = 0
    for i in range(len(loggers_for_flooding_per_horizon)):
        if (loggers_for_flooding_per_horizon[i].loc("PROBLEM", 0) == "FloodingSolver"):
            flood_i = i
            break
    loggers_for_flooding_per_horizon = loggers_for_flooding_per_horizon[flood_i].get_best_groupes_by("ACCURACY", "HORIZON", maximize=True)
    
    
    file = open("./_Artifacts/logs.txt", "w")
    loggers_per_problem.group_by("PROBLEM").render(file, "Best models")
    RUN_LOGGER.group_by("PROBLEM", inplace=False).render(file, "All runs")
    loggers_for_flooding_per_horizon.group_by("HORIZON").render(file, "FloodingSolver by horizon")
    file.close()