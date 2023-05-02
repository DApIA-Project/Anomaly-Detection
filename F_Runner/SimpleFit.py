

from _Utils.mlflow import init_ml_flow
from _Utils.module import module_to_dict



# for auto-completion
from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as _Trainer_


import mlflow

def simple_fit(Model:"type[_Model_]", Trainer:"type[_Trainer_]", CTX, project_name:str):
    """
    Just fit one time the model with the given hyperparameters

        Args:
            model:
                class of the model needed by the trainer

            trainer:
                used trainer

            CTX:
                the hyperparameters module (your constant.py file)
            
            project_name:
                the title for each mlflow experiment
                (eperiment name : project_name + "-" + model_name)
    """
    # desactivate warnings
    import warnings
    warnings.filterwarnings("ignore")
    

    run_number = init_ml_flow(project_name+"-"+Model.name, "genetic hyperparameters optimization of the model : "+Model.name)
    run_name = str(run_number) + " - " + Model.name

    CTX = module_to_dict(CTX)


    with mlflow.start_run(run_name=run_name) as run:
        # log the CTX
        for k in CTX:
            # if bool : convert to int
            if (type(CTX[k]) == bool):
                mlflow.log_param(k, int(CTX[k]))
            else:
                mlflow.log_param(k, CTX[k])
        
        # create the trainer
        trainer = Trainer(CTX, Model)
        # train the model
        loss_history, eval_metrics = trainer.run()

        print(eval_metrics)

        # log the metrics to mlflow
        for k in eval_metrics:
            mlflow.log_metric(k, eval_metrics[k])

