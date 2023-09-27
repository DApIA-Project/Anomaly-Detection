



# Mlflow logging
import _Utils.mlflow as mlflow

# Convert CTX to dict for logging hyperparameters
from _Utils.module import module_to_dict 

# For auto-completion, we use Abstract class as virtual type
from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as _Trainer_



def simple_fit(Model:"type[_Model_]", Trainer:"type[_Trainer_]", CTX, default_CTX=None, experiment_name:str = None):
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

    # Convert CTX to dict and log it
    CTX = module_to_dict(CTX)
    if (default_CTX != None):
        default_CTX = module_to_dict(default_CTX)
        for param in default_CTX:
            if (param not in CTX):
                CTX[param] = default_CTX[param]

    # test several CTX
    take_off = [True, False]
    map_context = [True, False]
    merge_labels = [
        { # merge 
            2: [1, 2, 3, 4, 5], # PLANE
            6: [6, 7, 10], # SMALL
            9: [9, 12], # HELICOPTER

            0: [8, 11] # not classified
        },
        {
            2: [1, 2, 4], # PLANE
            3: [3, 5],
            6: [6],
            7: [7],
            9: [9], # HELICOPTER
            10: [10],
            11: [11], # Military
            12: [12], # SAMU

            0: [8] # not classified
        }
    ]

    for to in take_off:
        for mc in map_context:
            for ml in merge_labels:

                SUB_CTX = CTX.copy()
                SUB_CTX["ADD_TAKE_OFF_CONTEXT"] = to
                SUB_CTX["ADD_MAP_CONTEXT"] = mc
                SUB_CTX["MERGE_LABELS"] = ml
                SUB_CTX["FEATURES_OUT"] = len(SUB_CTX["MERGE_LABELS"])-1
                SUB_CTX["USED_LABELS"] = [k for k in SUB_CTX["MERGE_LABELS"].keys() if k != 0]


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
                        mlflow.log_metric(metric_label, value)

