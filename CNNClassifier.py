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
class CNNModel(TwoInDNNModel):

    def __init__(self):
        TwoInDNNModel.__init__(self)
        self.name="TwoInputCNN"

    def buildModel(self):

        datashape = (initConfig.config["maxWords"], initConfig.config["features"])

        #word net
        input1=layers.Input(shape=datashape,name="em1")
        input2=layers.Input(shape=datashape,name="em2")

        comCNN = layers.Conv1D(filters=32,kernel_size=5,padding="same",activation="relu",
                               kernel_initializer=W_init,bias_initializer=b_init)
        comPool = layers.AveragePooling1D(pool_size=4)

        comCNN2 = layers.Conv1D(filters=64, kernel_size=5,padding="same", activation="relu",
                                kernel_initializer=W_init, bias_initializer=b_init)
        comPool2 = layers.AveragePooling1D(pool_size=4)

        comCNN3 = layers.Conv1D(filters=128, kernel_size=5, padding="same", activation="relu",
                                kernel_initializer=W_init, bias_initializer=b_init)
        comPool3 = layers.AveragePooling1D(pool_size=4)

        x1=comCNN(input1)
        x2=comCNN(input2)
        x1=comPool(x1)
        x2=comPool(x2)

        x1 = comCNN2(x1)
        x2 = comCNN2(x2)
        x1 = comPool2(x1)
        x2 = comPool2(x2)

        x1 = comCNN3(x1)
        x2 = comCNN3(x2)
        x1 = comPool3(x1)
        x2 = comPool3(x2)

        flatLayer=layers.Flatten()
        feature1=flatLayer(x1)
        feature2=flatLayer(x2)

        # sim net
        L1_distance = lambda x: K.abs(x[0] - x[1])
        both = layers.merge([feature1, feature2], mode=L1_distance, output_shape=lambda x: x[0])
        hiddenLayer=layers.Dense(units=1024,activation="relu",bias_initializer=b_init)(both)

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
    from FeatureDataSet import NLPDataSet
    from utilityFiles import splitTrainValidate
    from WordModel import WordEmbedding
    # embedding words
    emModel = WordEmbedding()
    emModel.loadModel()

    splitratio=0.9
    if splitratio>0 and splitratio<1:
        splitTrainValidate("../data/train_nlp_data.csv",splitratio)

        trainData=getFeedData("../data/train.csv",emModel)
        validateData=getFeedData("../data/validate.csv",emModel)
        dnnmodel = CNNModel()

        dnnmodel.trainModel(trainData, validateData)
        dnnmodel.saveModel()
        exit(1)
    else:
        trainData,validateData=getFeedData("../data/train_nlp_data.csv",emModel),None

    model_num = initConfig.config["cnnNum"]
    dataList = trainData.getFold(model_num)
    for i in range(model_num):
        # cnn dnn model
        dnnmodel = CNNModel()

        dnnmodel.name += str(i)
        train, test = dataList[i]
        dnnmodel.trainModel(train, test)
        dnnmodel.saveModel()

        print("\n==========%d/%d=================\n"%(i+1,model_num))

    exit(2)


