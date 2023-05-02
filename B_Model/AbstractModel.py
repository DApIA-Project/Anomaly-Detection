

import numpy as np


class Model():
    """
    Abstrac Class representing a model
    /!\ your model must always contain all the following methods /!\
    (but not inherit from this class)
    
    Methods :
    ---------

    predict(x):
        return the prediction of the model

    compute_loss(x, y):
        return the loss and the prediction associated to x, y and y_

    training_step(x, y):
        do one training step.
        return the loss and the prediction of the model for this batch
    """

    name = "AbstractModel"

    def __init__(self, CTX:dict, name="Model"):
        """ 
        Generate model architecture
        Define loss function
        Define optimizer
        """
        pass

    def predict(self, x):
        """
        Make prediction for x (comming from the dataloader)

        you can use this function to make the 
        conversion from the format needed by the model
        and the format given by the dataloader
        """
        
        raise NotImplementedError

    # call directly the layers of the model
    # only usefull for printing the model with tensorflow
    def __raw_call__(self, x):
        raise NotImplementedError
        
    # call the model with all tensorflow optimisation and stuff
    def __call__(self, x):
        return self.__raw_call__(x)

    def compute_loss(self, x, y):
        raise NotImplementedError
        y_ = self.predict(x)
        return 0.0, y_

    def training_step(self, x, y):
        raise NotImplementedError
        loss, out = self.compute_loss(x, y)
        # compute and apply gradient
        return loss, out


    def visualize(self, filename="./Output_artefact/model.png"):
        """
        Generate a visualization of the model
        """
        raise NotImplementedError
        input_shape=(0, 0, 0)
    
        # get a keras.plot_model image
        input = tf.keras.Input(shape=input_shape, dtype='int32', name='input')
        output = self.__call__(input)
        
        model = tf.keras.Model(inputs=[input], outputs=[output])
        tf.keras.utils.plot_model(model, to_file=filename, show_shapes=True)