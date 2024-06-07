

# Import the model
from B_Model.ReplaySolver.HASH import Model

# Import the context (hyperparameters, constants, etc...)
import C_Constants.ReplaySolver.HASH as CTX
import C_Constants.ReplaySolver.DefaultCTX as DefaultCTX

# Import the training loop adapted to the model
from E_Trainer.ReplaySolver.Trainer import Trainer as HashTrainer

# Choose the training method
from F_Runner.FitOnce import fitOnce
from F_Runner.MultiFit import multiFit

from _Utils.os_wrapper import os


def __main__() -> None:
    parent_dir = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    fitOnce(Model, HashTrainer, CTX, default_CTX=DefaultCTX, experiment_name=parent_dir)

