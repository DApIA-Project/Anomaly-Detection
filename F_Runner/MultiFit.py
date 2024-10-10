

# Convert CTX to dict for logging hyperparameters
from _Utils.module import buildCTX
from numpy_typing import np, ax, ax
from   _Utils.Color import prntC
import _Utils.Color as C

# For auto-completion, we use Abstract class as virtual type
from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as _Trainer_


def multiFit(Model:"type[_Model_]", Trainer:"type[_Trainer_]", CTX, default_CTX=None, repeats=2, experiment_name:str = None):
    """
    Fit the model several times with a given set of hyperparameters
    to check the stability of the training.

    Parameters:
    -----------
    model: type[Model]:
        Model used for training

    trainer: type[Trainer]
        Trainer class, managing the training loop, testing and evaluation, for a specific task
        (eg. spoofing detection)

    CTX: Module
        Python module containing the set of hyperparameters

    repeats: int
        Number of times the model is trained with the same hyperparameters
    """

    # Convert CTX to dict and merge it with default_CTX
    CTX = buildCTX(CTX, default_CTX)


    metrics_stats:dict[str,list] = {}
    for i in range(repeats):


        # Create a new training environment and run it
        trainer = Trainer(CTX, Model)
        metrics = trainer.run()

        # save the results
        for metric_label in metrics:
            value = metrics[metric_label]
            if (metric_label not in metrics_stats):
                metrics_stats[metric_label] = [value]
            else:
                metrics_stats[metric_label].append(value)

        # Analyze the results if it is the last run
        if (i == repeats -1):
            prntC("Metric".rjust(15), "|", "min".rjust(10), "|", "mean".rjust(10), "|", "std".rjust(10), "|", "median".rjust(10), "|", "max".rjust(10), sep="")
            for metric in metrics:
                # min, mean, std, median, max
                _min = np.min(metrics_stats[metric])
                _mean = np.mean(metrics_stats[metric])
                _std = np.std(metrics_stats[metric])
                _median = np.median(metrics_stats[metric])
                _max = np.max(metrics_stats[metric])

                prntC(metric.rjust(15), "|", str(round(_min, 3)).rjust(10), "|", str(round(_mean, 3)).rjust(10), "|", str(round(_std, 3)).rjust(10), "|", str(round(_median, 3)).rjust(10), "|", str(round(_max, 3)).rjust(10), sep="")


