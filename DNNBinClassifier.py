from keras import models,layers,optimizers,losses
import time,json,keras.backend as K
from keras import utils
from sklearn import metrics
import warnings
import numpy as np
from initConfig import config
warnings.filterwarnings("ignore")

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



#create model
def createDNN(dp=0.5):
    inputDim=2*config["features"]
    ouputDim=2
    DNNmodel=models.Sequential()
    DNNmodel.add(layers.Dense(units=96,input_shape=(inputDim,),activation="relu"))
    DNNmodel.add(layers.Dense(units=96,activation="relu"))
    DNNmodel.add(layers.Dense(units=64,activation="relu"))
    DNNmodel.add(layers.Dense(units=64,activation="relu"))
    DNNmodel.add(layers.Dense(units=32,activation="relu"))
    DNNmodel.add(layers.Dense(units=32,activation="relu"))
    DNNmodel.add(layers.Dense(units=16,activation="relu"))
    DNNmodel.add(layers.Dropout(dp))
    DNNmodel.add(layers.Dense(units=ouputDim,activation="softmax"))

    opt = optimizers.Adadelta()
    DNNmodel.compile(optimizer=opt,loss=losses.binary_crossentropy)
    return DNNmodel

#model
class DNNCLassifier:

    def __init__(self):
        self.name="DNNBinaryClassifier"
        self.params={

            'dp':0.8,
            'verbose':0,
        }


    def trainModel(self,dataSet):
        print(self.name+" training")
        t0=time.time()

        self.model=createDNN(self.params["dp"])

        self.model.fit(dataSet.trainX,utils.to_categorical(dataSet.trainY,2),verbose=1,epochs=5,batch_size=500)

        t1=time.time()
        y_predict=self.predict(dataSet.trainX)
        #print(dataSet.validateY[:3])
        #print(y_predict[:3])

        f1=metrics.f1_score(dataSet.trainY,y_predict)

        print("finished in %ds"%(t1-t0),"f1=",f1)

    def predict(self,X):

        Y=self.model.predict(X,verbose=0)
        Y=np.argmax(Y,axis=1)
        return Y

    def loadModel(self):
        self.model=models.load_model("./models/" + self.name + ".h5")
    def saveModel(self):
        self.model.save("./models/" + self.name + ".h5")
