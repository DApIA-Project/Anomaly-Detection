

# Import the model
from B_Model.FloodingSolver.CNN import Model

# Import the context (hyperparameters, constants, etc...)
import C_Constants.FloodingSolver.CNN as CTX
import C_Constants.FloodingSolver.DefaultCTX as DefaultCTX

# Import the training loop adapted to the model
from E_Trainer.FloodingSolver.Trainer import Trainer

# Choose the training method
from F_Runner.FitOnce import fitOnce
from F_Runner.MultiFit import multiFit

import os



def __main__() -> None:
    parent_dir = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    fitOnce(Model, Trainer, CTX, default_CTX=DefaultCTX, experiment_name=parent_dir)

