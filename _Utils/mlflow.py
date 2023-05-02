import mlflow
import os


def init_ml_flow(experiments_name, experiments_desc):

    os.environ["MLFLOW_TRACKING_URI"] = "http://51.77.221.41:8000"


    # if experiment does not exist, create it
    if mlflow.get_experiment_by_name(experiments_name) is None:
        mlflow.create_experiment(experiments_name)
    mlflow.set_experiment(experiments_name)

    # the run name is formated as folow :
    # ${run_number} - ${run_desc}
    # if run_desc is empty, the run name is formated as folow :
    # ${run_number}

    # get the run with the highest timestamp
    last_run = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(experiments_name).experiment_id)
    run_number = 1
    if not(last_run.empty):
        run_name = last_run.iloc[0]["tags.mlflow.runName"]
        print(run_name)
        if run_name.find("-") == -1:
            run_number = int(run_name) + 1
        else:
            run_number = int(run_name.split("-")[0]) + 1
    
    return run_number
