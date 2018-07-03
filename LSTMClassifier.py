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
    from WordModel import WordEmbedding
    from FeatureDataSet import NLPDataSet
    from utilityFiles import splitTrainValidate
    from imblearn import over_sampling
    # embedding words
    emModel = WordEmbedding()
    emModel.loadModel()

    splitratio=1
    if splitratio>0 and splitratio<1:
        splitTrainValidate("../data/train_nlp_data.csv",splitratio)

        trainData=getFeedData("../data/train.csv",emModel)

        # resample methods
        '''
        datashape = (initConfig.config["maxWords"], initConfig.config["features"])

        X,Y=np.concatenate((trainData.dataEm1,trainData.dataEm2),axis=1),trainData.dataY
        X=np.reshape(X,newshape=(len(X),2*datashape[0]*datashape[1]))
        osam=over_sampling.ADASYN(n_jobs=10)
        X,Y=osam.fit_sample(X,Y)

        trainData.dataEm1,trainData.dataEm2,trainData.dataY=\
            np.reshape(X[:,:datashape[0]*datashape[1]],newshape=(len(X),datashape[0],datashape[1])),\
            np.reshape(X[:,datashape[0]*datashape[1]:],newshape=(len(X),datashape[0],datashape[1])),\
            Y
        '''


        validateData=getFeedData("../data/validate.csv",emModel)
    else:
        trainData,validateData=getFeedData("../data/train_nlp_data.csv",emModel),None

    model_num=initConfig.config["lstmNum"]
    dataList=trainData.getFold(model_num)
    for i in range(model_num):
        # lstm dnn model
        dnnmodel = TwoInDNNModel()

        dnnmodel.name+=str(i)
        train,test=dataList[i]
        dnnmodel.trainModel(train,test)
        dnnmodel.saveModel()

        print("\n==========%d/%d=================\n" % (i + 1, model_num))