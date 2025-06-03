

# Import the model
from B_Model.AircraftClassification.CNN_V1 import Model

# Import the context (hyperparameters, constants, etc...)
import C_Constants.AircraftClassification.CNN_V1 as CTX
import C_Constants.AircraftClassification.DefaultCTX as DefaultCTX

# Import the training loop adapted to the model
from E_Trainer.AircraftClassification.Trainer import Trainer

# Choose the training method
from F_Runner.FitOnce import fitOnce
from F_Runner.MultiFit import multiFit


from _Utils.os_wrapper import os



def __main__() -> None:
    parent_dir = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    fitOnce(Model, Trainer, CTX, default_CTX=DefaultCTX, experiment_name=parent_dir)

    multiFit(Model, Trainer, CTX, default_CTX=DefaultCTX, experiment_name=parent_dir,
             tested_values={
                "ADD_TAKE_OFF_CONTEXT": [True] * 5 + [False] * 5,
             })