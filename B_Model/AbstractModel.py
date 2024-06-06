

from _Utils.numpy import np



class Model():

    name = "AbstractModel (TO OVERRIDE)"

    def __init__(self, CTX:dict) -> None:
        """
        Generate model architecture
        Define loss function
        Define optimizer
        """
        pass


    def predict(self, x:np.ndarray) -> np.ndarray:
        """
        Make prediction for x
        """
        raise NotImplementedError

    def compute_loss(self, x:np.ndarray, y:np.ndarray) -> "tuple[float, np.ndarray]":
        """
        Make a prediction and compute the loss
        that will be used for training
        """
        raise NotImplementedError

    def training_step(self, x:np.ndarray, y:np.ndarray) -> "tuple[float, np.ndarray]":
        """
        Do one forward pass and gradient descent
        for the given batch
        """
        raise NotImplementedError


    def visualize(self, save_path:str="./_Artifacts/") -> None:
        """
        Generate a visualization of the model's architecture
        """
        pass



    def get_variables(self)->np.ndarray:
        """
        Return the variables of the model
        """
        raise NotImplementedError
    def set_variables(self, variables:object) -> None:
        """
        Set the variables of the model
        """
        raise NotImplementedError
