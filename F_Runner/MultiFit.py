
from _Utils.DebugGui import launch_gui

# Convert CTX to dict for logging hyperparameters
from _Utils.module import buildCTX
from numpy_typing import np, ax

# For auto-completion, we use Abstract class as virtual type
from B_Model.AbstractModel import Model as _Model_
from E_Trainer.AbstractTrainer import Trainer as _Trainer_
from F_Runner.Utils import log_data



def multiFit(Model:"type[_Model_]", Trainer:"type[_Trainer_]", CTX, default_CTX=None, experiment_name:str = None, tested_values:"dict[str, list[object]]" = {}):


    # Convert CTX to dict and merge it with default_CTX
    BASE_CTX = buildCTX(CTX, default_CTX)
    
    actual_test = [0 for i in range(len(tested_values))]
    nb_test = np.prod([len(tested_values[k]) for k in tested_values])
    
    
    for t in range(nb_test):
        
        # build CTX from actual_test
        CTX = BASE_CTX.copy()
        for i, k in enumerate(tested_values):
            # get the index of the current test
            index = actual_test[i]
            # get the value of the current test
            value = tested_values[k][index]
            # set the value in CTX
            CTX[k] = value
            
        

        # Create a new training environment and run it
        launch_gui(CTX)
        trainer = Trainer(CTX, Model)
        
        metrics = trainer.run()
        
        log_data(metrics, CTX, Model, experiment_name=experiment_name)
       
        # create the next test
        for i, k in enumerate(tested_values):
            max_value = len(tested_values[k]) - 1
            if (actual_test[i] < max_value):
                actual_test[i] += 1
                break
            else:
                actual_test[i] = 0
            
    
    
    


