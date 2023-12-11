import os


# To deactivate mlflow logging, set USE_MLFLOW to False
#   (this is usefull when you want to run the code on a computer without mlflow installed)

USE_MLFLOW = False



if (USE_MLFLOW):

    # Import all mlflow functions (to make them available in the whole project)
    from mlflow import *


    def init_ml_flow(experiments_name):
        """
        Init the mlflow logging

        Parameters:
        -----------

        experiments_name: str
            mlflow experiment name

        Returns:
        --------

        run_number: int
            The n-th value of the last run of the experiment
        """

        if (USE_MLFLOW):
            os.environ["MLFLOW_TRACKING_URI"] = "http://51.77.221.41:8000"


            # If experiment does not exist, create it
            if get_experiment_by_name(experiments_name) is None:
                create_experiment(experiments_name)
            set_experiment(experiments_name)

            # The run name is formated as folow :
            # ${run_number} - ${run_desc} or ${run_number}

            # Get the run with the highest timestamp
            last_run = search_runs(experiment_ids=get_experiment_by_name(experiments_name).experiment_id)
            
            # Extract the run number
            run_number = 1
            if not(last_run.empty):
                run_name = last_run.iloc[0]["tags.mlflow.runName"]
                if run_name.find("-") == -1:
                    run_number = int(run_name) + 1
                else:
                    run_number = int(run_name.split("-")[0]) + 1    
            return run_number
        
        return 0


# If mlflow is deactivated, we create dummy functions
# to avoid errors when trainers, models, ... are calling mlflow logging functions
else:
    def init_ml_flow(experiments_name):
        return 0

    def log_metric(key: str,
        value: float,
        step: int = None):
        pass
    
    # Dummy class for the syntax : with mlflow.start_run() as run:
    class __DUMMY_WITH__:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass
    
    # Start run usable in a with statement
    def start_run(run_id: str = None,
        experiment_id: str= None,
        run_name: str= None,
        nested: bool = False,
        tags = None,
        description: str = None):

        return __DUMMY_WITH__("")

    def log_param(key: str, value):
        pass