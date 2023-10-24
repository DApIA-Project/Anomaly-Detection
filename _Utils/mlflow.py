import os


# To deactivate mlflow logging, set USE_MLFLOW to False
#   (this is usefull when you want to run the code on a computer without mlflow installed)

USE_MLFLOW = False



if (USE_MLFLOW):
    pass

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