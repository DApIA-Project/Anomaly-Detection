

# Import the model
from B_Model.FloodingSolver.LSTM import Model

# Import the context (hyperparameters, constants, etc...)
import C_Constants.FloodingSolver.LSTM as CTX
import C_Constants.FloodingSolver.DefaultCTX as DefaultCTX

# Import the training loop adapted to the model
from E_Trainer.FloodingSolver.Trainer import Trainer

# Choose the training method
from F_Runner.FitOnce import fitOnce
from F_Runner.MultiFit import multiFit

from _Utils.os_wrapper import os



def __main__() -> None:
    parent_dir = os.path.dirname(os.path.abspath(__file__)).split("/")[-1]
    # multiFit(Model, Trainer, CTX, default_CTX=DefaultCTX, experiment_name=parent_dir,
    #          tested_values={
    #             #  "HORIZON": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    #              "HORIZON": [22],
    #          })

    fitOnce(Model, Trainer, CTX, default_CTX=DefaultCTX, experiment_name=parent_dir)

