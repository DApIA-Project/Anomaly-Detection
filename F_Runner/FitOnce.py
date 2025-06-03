
from _Utils.DebugGui import launch_gui

# Convert CTX to dict for logging hyperparameters
from _Utils.module import buildCTX
from numpy_typing import np, ax

# For auto-completion, we use Abstract class as virtual type
from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as _Trainer_
from F_Runner.Utils import log_data



def fitOnce(Model:"type[_Model_]", Trainer:"type[_Trainer_]", CTX, default_CTX=None, experiment_name:str = None):

    # Convert CTX to dict and merge it with default_CTX
    CTX = buildCTX(CTX, default_CTX)

    # Create a new training environment and run it
    launch_gui(CTX)
    trainer = Trainer(CTX, Model)
    
    metrics = trainer.run()
    
    log_data(metrics, CTX, Model, experiment_name=experiment_name)

    
    
    
    
    


