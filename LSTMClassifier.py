from keras import models,layers,optimizers,losses,metrics
import time
from keras import utils
import warnings
import numpy as np
import initConfig
import keras.backend as K
warnings.filterwarnings("ignore")
from DNNModel import *
#model
class LSTMModel(TwoInDNNModel):

    def __init__(self):
        TwoInDNNModel.__init__(self)
        self.name="TwoInputLSTM"
        self.numEpoch=10
    def buildModel(self):
        datashape=(initConfig.config["maxWords"],initConfig.config["features"])

        #word net
        input1=layers.Input(shape=datashape,name="em1")
        input2=layers.Input(shape=datashape,name="em2")
        comLSTM=layers.LSTM(128,kernel_initializer=W_init)
        encode1=comLSTM(input1)
        encode2=comLSTM(input2)

        # extract net2

        L1_distance = lambda x: K.abs(x[0] - x[1])
        both = layers.merge([encode1, encode2], mode=L1_distance, output_shape=lambda x: x[0])

        hiddenLayer=layers.Dense(units=64,activation="relu",bias_initializer=b_init)(both)
        dropLayer=layers.Dropout(0.36)(hiddenLayer)
        predictionLayer=layers.Dense(units=2,name="label",activation="softmax")(dropLayer)
        self.model=models.Model(inputs=[input1,input2],
                                outputs=[
                                    predictionLayer,
                                ]
                                )

        self.model.compile(optimizer=optimizers.Adam(),
                      loss={
                          "label":losses.binary_crossentropy
                           }
                      )

        return self.model


if __name__ == '__main__':
    trainModel(LSTMModel)