 
from _Utils.mlflow import init_ml_flow
from _Utils.module import module_to_dict

from _Utils.mutation_func import *

from B_Model.AbstractModel import Model as _Model_ # for auto-completion
from E_Trainer.AbstractTrainer import Trainer as _Trainer_ # for auto-completion


import mlflow
import numpy as np


__ctx_history__ = []
__last_mutation_transform__ = (None, None)
__last_ctx__ = None

def __mutateCTX__(bests_CTX:"list[dict]", traiable_params:"dict[str, Mutation]", same=False):
    """
    generate a new variation of a CTX, which has never been tested before

        Args:
            bests_CTX: 
                a sorted list of the best hyperparameters found

            traiable_params: 
                a dict of the hyperparameters which 
                as been alowed to be trained by the user

        Returns:
            dict: 
                a new, unique CTX
    """
    # get access to the history of tested CTX
    global __ctx_history__, __last_mutation_transform__, __last_ctx__

    hyperparameters_names = list(traiable_params.keys())

    ctx_i = 0
    ctx = bests_CTX[0].copy()

    n = 0
    while (ctx in __ctx_history__):
        # select a random good hyperparameter context from the best database
        ctx = __last_ctx__
        if not(same):
            ctx_i = np.random.randint(len(bests_CTX))
            ctx = bests_CTX[ctx_i].copy()

        # select a random hyperparameter to mutate
        mutated_param = __last_mutation_transform__[0]
        direction = __last_mutation_transform__[1]
        if not(same):
            mutated_param = np.random.choice(hyperparameters_names)
            direction = np.random.choice([-2, -1, 1, 2])

        # get the type of the variable (int or float) to avoid having 
        # --> float epoch number
        # --> int learning rate
        # configure the mutation
        mutated_param_type = type(ctx[mutated_param])
        mutation_function = traiable_params[mutated_param].function
        mutation_rate = traiable_params[mutated_param].rate
        mutation_min = traiable_params[mutated_param].min
        mutation_max = traiable_params[mutated_param].max
        mutated_check = traiable_params[mutated_param].check

        # mutate the hyperparameter

        ctx[mutated_param] = mutation_function(ctx[mutated_param], direction, mutation_rate)

        
        # check range
        if (mutation_min != None):
            if (ctx[mutated_param] < mutation_min):
                ctx[mutated_param] = mutation_min
        if (mutation_max != None):
            if (ctx[mutated_param] > mutation_max):
                ctx[mutated_param] = mutation_max

        if (mutated_check != None):
            if (not mutated_check(ctx)):
                # if the mutation is not valid,
                # we try again with a new CTX
                print("Invalid CTX state !!! -> redo the mutation")
                ctx = bests_CTX[0].copy()
                continue

        # convert back to the original type
        ctx[mutated_param] = mutated_param_type(ctx[mutated_param])
        if (mutated_param_type == float):
            ctx[mutated_param] = round(ctx[mutated_param], 5)

        if not(same):
            __last_mutation_transform__ = (mutated_param, abs(direction)//direction)
            __last_ctx__ = ctx.copy()
        else:
            same = False
        
        print("Mutate : ", mutated_param, ", from : ", bests_CTX[ctx_i][mutated_param], ", to : ", ctx[mutated_param])

        # count the number of iteration done to 
        # stop the loop if we are unable to find a new CTX
        n += 1
        if (n > 1000):
            print("no more CTX to test")
            return None

    __ctx_history__.append(ctx)

    return ctx










def genetic_fit(Model:"type[_Model_]", Trainer:"type[_Trainer_]", CTX, traiable_params:"dict[str, list[function, float, bool]]", evaluated_on:str, project_name:str):
    """
    loop over various hyperparameters to find the best one

        Args:
            model:
                class of the model needed by the trainer

            trainer:
                used trainer

            CTX:
                the hyperparameters module (your constant.py file)
        
            traiable_params:
                The dict of hyperparameters alowed to be trained
                Contain also the way to mutate them.

            evaluated_on:
                the metric (returned by the trainer) 
                to use to evaluate the hyperparameters set

            project_name:
                the title for each mlflow experiment
                (eperiment name : project_name + "-" + model_name)


        Returns:
            The best hyperparameters found
    """
    # desactivate warnings
    import warnings
    warnings.filterwarnings("ignore")


    # nb iteration, nb bests to keep
    NB_RUNS = 400
    MAX_BESTS = 3
    ITER = 5

    # init mlflow
    run_number = init_ml_flow(project_name+"-"+Model.name, "genetic hyperparameters optimization of the model : "+Model.name)
    
    # generate the constant dict from the constant file
    # list all variables in the ctx namespace
    CTX = module_to_dict(CTX)

    # check if all the hyperparameters choosed by the user are in the CTX
    for k in traiable_params:
        if k not in CTX:
            raise Exception("'traiable_params' contain the key : '" + k + "' which is not in the CTX")


    # init the best CTX with the default one
    best_CTX = [CTX.copy()]
    best_score = [100000]
    worked = False

    # run all the tests
    for t in range(NB_RUNS):
        
        # generate new CTX
        CTX = __mutateCTX__(best_CTX, traiable_params, same=worked)

        if (CTX is None):
            break

        # start a new mlflow run :
        run_name = str(run_number) + " - " + Model.name
        if (NB_RUNS > 0):
            run_name += "(" + str(t + 1)+")"

        with mlflow.start_run(run_name=run_name) as run:
            # log the new mutated CTX
            for k in CTX:
                # if bool : convert to int
                if (type(CTX[k]) == bool):
                    mlflow.log_param(k, int(CTX[k]))
                else:
                    mlflow.log_param(k, CTX[k])

            eval_metrics_list = []
            for iter in range(ITER):
                # create the trainer
                trainer = Trainer(CTX, Model, iter)
                loss_history, eval_metrics = trainer.run()
                eval_metrics_list.append(eval_metrics)

            print(eval_metrics_list)

            # average the metrics
            eval_metrics = {}
            for k in eval_metrics_list[0]:
                eval_metrics[k] = 0
                for i in range(ITER):
                    eval_metrics[k] += eval_metrics_list[i][k]
                eval_metrics[k] /= ITER
        

            # insert the new best CTX based on the evaluated_on metric
            i = 0
            while (eval_metrics[evaluated_on] > best_score[i]):
                i += 1
                if (i >= MAX_BESTS):
                    break
            best_CTX.insert(i, CTX.copy())
            best_score.insert(i, eval_metrics[evaluated_on])

            # remove the last one if there is too much
            if (len(best_CTX) > MAX_BESTS):
                best_CTX.pop()
                best_score.pop()

                # if last best is full and i is not the last one, save this mutation and repeat it
                if (i < MAX_BESTS - 1):
                    worked = True
                else:
                    worked = False

            # log the best score CTX
            print("Best score : ", best_score)


            # log the metrics to mlflow
            for k in eval_metrics:
                mlflow.log_metric(k, eval_metrics[k])


        
            





   






















# utils


class Mutation:
    def __init__(self, function, rate=None, min=None, max=None, check=None):
        self.function = function
        self.rate = rate
        self.min = min
        self.max = max
        self.check = check