
from _Utils.DebugGui import launch_gui
import _Utils.FeatureGetter as FG

# Convert CTX to dict for logging hyperparameters
from _Utils.module import module_to_dict
from _Utils.numpy import np, ax

# For auto-completion, we use Abstract class as virtual type
from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as _Trainer_


def fitOnce(Model:"type[_Model_]", Trainer:"type[_Trainer_]", CTX, default_CTX=None, experiment_name:str = None):
    """
    Fit the model once with a given set of hyperparameters.

    Parameters:
    -----------
    model: type[Model]:
        Model used for training

    trainer: type[Trainer]
        Trainer class, managing the training loop, testing and evaluation, for a specific task
        (eg. spoofing detection)

    CTX: Module
        Python module containing the set of hyperparameters
    """

    # Convert CTX to dict and merge it with default_CTX
    CTX = module_to_dict(CTX)
    if (default_CTX != None):
        default_CTX = module_to_dict(default_CTX)
        for param in default_CTX:
            if (param not in CTX):
                CTX[param] = default_CTX[param]

    FG.init(CTX)

    # Create a new training environment and run it
    launch_gui(CTX)
    trainer = Trainer(CTX, Model)
    metrics = trainer.run()


